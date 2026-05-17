import json
import pathlib
import torch
from argparse import ArgumentParser

from wav2vec2.model_1_01_split_save import (
    wav2vec2_model_split_save, wav2vec2_model, wav2vec2_model_split_half
)

attn_mask = None

def set_tau_min_max(model, eta_min=0.01, eta_max=0.5):
    def apply_tau_min_max(module):
        if hasattr(module, 'eta_min'):
            module.eta_min = eta_min
            module.eta_max = eta_max
    
    for name, module in model.named_modules():
        # if isinstance(module, Gate):
        apply_tau_min_max(module)
    
    return model

def set_tau_reverse(model, reverse_tau=None):
    def apply_tau_reverse(module):
        if hasattr(module, 'reverse_tau'):
            module.reverse_tau = reverse_tau
    
    for name, module in model.named_modules():
        apply_tau_reverse(module)
    
    return model

def set_shrink_scale(model, shrink_scale=None):
    def apply_tau_reverse(module):
        if hasattr(module, 'shrink_scale'):
            module.shrink_scale = shrink_scale
    
    for name, module in model.named_modules():
        apply_tau_reverse(module)
    
    return model

def set_attn_masks_for_save(model, attn_mask=None):
    num_qk_list = [d['num_qk'] for d in attn_mask]
    num_vo_list = [d['num_vo'] for d in attn_mask]
    
    def apply_attn_masks(module, qk_unmask_num, vo_unmask_num):
        
        if hasattr(module, 'qk_unmask_num'):
            module.qk_unmask_num = qk_unmask_num
        if hasattr(module, 'vo_unmask_num'):
            module.vo_unmask_num = vo_unmask_num
    layer_index = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'qk_unmask_num'):
            apply_attn_masks(module, num_qk_list[layer_index], num_vo_list[layer_index])
            layer_index+=1
    
    assert layer_index == 12
    
    return model

def prune_from_ckpt(distilled_ckpt, original_ckpt):
    ckpt = torch.load(distilled_ckpt, map_location='cpu')
    student_model_state_dict = {
        k[len("student_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")
    }
    distill_linear_projs_state_dict = {
        k[len("distill_linear_projs."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("distill_linear_projs.")
    }
    # stu_ckpt = torch.load('/project_bdda8/bdda/hnxu/prune/SSLprune/DPHuBERT_pretrain/exp_minmax_split/hubert-base_train100_sp0.75_spup5000_lr0.0002_up15000_max50000_layer2layer0.4,8,12_reglr0.02_gatelr1e-4_head,interm_sec320_accum2_gate_norm_0.5_0.01_tau_gate_up_30%_down_re_split_new_forward/lr0.0001_up5000_max25000/ckpts/pruned_hubert_base.pth', map_location='cpu')
    # config = stu_ckpt['config']
    config = torch.load(original_ckpt, map_location='cpu')['config']
    # print('config[encoder_num_heads]:',config['encoder_num_heads'])
    config['encoder_num_heads'] = [{'num_qk': 768, 'num_vo': 768} for _ in range(12)]
    # config['num_heads'] = [{'num_qk': 768, 'num_vo': 768}*12]
    # import pdb;pdb.set_trace()
    print('config:', config)
    config.update(
        dict(
            extractor_prune_conv_channels="feature_extractor.conv_layers.0.gate.gate_threshold" in student_model_state_dict,
            # encoder_prune_qk_heads="encoder.transformer.layers.0.attention.gate_for_qk.gate_threshold" in student_model_state_dict,
            # encoder_prune_vo_heads="encoder.transformer.layers.0.attention.gate_for_vo.gate_threshold" in student_model_state_dict,
            encoder_prune_attention_heads="encoder.transformer.layers.0.attention.gate_for_vo.gate_threshold" in student_model_state_dict,
            encoder_prune_attention_layer="encoder.transformer.layers.0.attention.gate_for_layer.gate_threshold" in student_model_state_dict,
            encoder_prune_feed_forward_intermediate="encoder.transformer.layers.0.feed_forward.gate_for_intermediate.gate_threshold" in student_model_state_dict,
            encoder_prune_feed_forward_layer="encoder.transformer.layers.0.feed_forward.gate_for_layer.gate_threshold" in student_model_state_dict,
        )
    )
    # import pdb;pdb.set_trace()
    model = wav2vec2_model_split_save(**config)
    # print(student_model_state_dict["encoder.transformer.layers.0.attention.gate_for_vo.gate_threshold"])
    # student_model.encoder.transformer.layers[0].attention.gate_for_vo.gate_threshold
    # import pdb;pdb.set_trace()
    # student_model.encoder.transformer.layers[0].attention.final_distill
    model.load_state_dict(student_model_state_dict, strict=False)
    """Unexpected key(s) in state_dict: "encoder.transformer.layers.0.attention.gate_for_qk.gate_threshold"""
    
    # model.load_state_dict(student_model_state_dict)
    model = set_tau_min_max(model, eta_min=args.eta_min, eta_max=args.eta_max)
    model = set_tau_reverse(model, reverse_tau=args.reverse_tau)
    model = set_shrink_scale(model, shrink_scale=args.shrink_scale)
    # conv_config, use_attention, use_feed_forward, num_heads, remaining_qk_heads, remaining_vo_heads, ff_interm_features = model.prune_split_save()
    # import pdb;pdb.set_trace()
    conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = model.prune()
    
    
    # import pdb;pdb.set_trace()
    pruned_config = config.copy()
    if len(num_heads) == 0:     # for wavlm
        assert len(remaining_heads) > 0
        pruned_config.update(
            {
                "encoder_remaining_heads": remaining_heads,
            }
        )
    else:
        pruned_config.update(
            {
                "encoder_num_heads": num_heads,
            }
        )
    pruned_config.update(
        {
            "extractor_conv_layer_config": conv_config,
            "encoder_use_attention": use_attention,
            "encoder_use_feed_forward": use_feed_forward,
            "encoder_ff_interm_features": ff_interm_features,
            "extractor_prune_conv_channels": False,
            "encoder_prune_attention_heads": False,
            "encoder_prune_attention_layer": False,
            "encoder_prune_feed_forward_intermediate": False,
            "encoder_prune_feed_forward_layer": False,
        }
    )
    # attn_mask = pruned_config["encoder_num_heads"]
    # pruned_config["encoder_num_heads"] = [12] * 12
    # import pdb;pdb.set_trace()
    # model = wav2vec2_model_split_save(**pruned_config)

    print(json.dumps(pruned_config, indent=4))
    
    
    ret = {
        "state_dict": model.state_dict(),
        "config": pruned_config,
        "distill_linear_projs": distill_linear_projs_state_dict,
        "attn_mask":attn_mask
    }
    return ret, attn_mask


def load_pruned_model(ckpt_path, attn_mask=None):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # attn_mask = ckpt["config"]["encoder_num_heads"]
    # ckpt["config"]["encoder_num_heads"] = [12] * 12
    # import pdb;pdb.set_trace()
    model = wav2vec2_model_split_save(**ckpt["config"])
    # model = set_attn_masks(model, attn_mask=attn_mask)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def parse_args():
    parser = ArgumentParser(description="Prune and save distilled model.")
    parser.add_argument(
        "--distilled_ckpt",
        type=pathlib.Path,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--shrink_scale",
        default=False,
        type=bool,
        help="eta_min for tau",
    )
    parser.add_argument(
        "--reverse_tau",
        default=False,
        type=bool,
        help="eta_min for tau",
    )
    parser.add_argument(
        "--eta_min",
        default=0.01,
        type=float,
        help="eta_min for tau",
    )
    parser.add_argument(
        "--eta_max",
        default=0.5,
        type=float,
        help="eta_max for tau",
    )
    parser.add_argument(
        "--original_ckpt",
        type=pathlib.Path,
        help="Path to the original checkpoint."
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    out_path = args.distilled_ckpt.parent / "pruned_hubert_base_stage1_split_save.pth"
    ret, attn_mask = prune_from_ckpt(
            distilled_ckpt=args.distilled_ckpt,
            original_ckpt=args.original_ckpt,
        )
    torch.save(
        ret,
        out_path
    )

    # Check if loading from ckpt works
    load_pruned_model(out_path, attn_mask)

    print(f"Successfully saved pruned model weights and config to: {out_path}")
