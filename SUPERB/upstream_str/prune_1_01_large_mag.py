import json
import pathlib
import torch
from argparse import ArgumentParser

from wav2vec2.model_1_01_mag import (
    wav2vec2_model,
)
# 2025.5.23 11:23 modified

# def set_norm_type(model, norm_type='minmax'):
#     def apply_norm_type(module):
#         if hasattr(module, 'norm_type'):
#             module.norm_type = norm_type
    
#     for name, module in model.named_modules():
#         apply_norm_type(module)
    
#     return model
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

def prune_from_ckpt(distilled_ckpt, original_ckpt):
    ckpt = torch.load(distilled_ckpt, map_location='cpu')
    student_model_state_dict = {
        k[len("student_model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")
    }
    distill_linear_projs_state_dict = {
        k[len("distill_linear_projs."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("distill_linear_projs.")
    }
    config = torch.load(original_ckpt, map_location='cpu')['config']
    config.update(
        dict(
            extractor_prune_conv_channels="feature_extractor.conv_layers.0.gate_for_conv.gate_threshold" in student_model_state_dict,
            encoder_prune_attention_heads="encoder.transformer.layers.0.attention.gate_for_heads.gate_threshold" in student_model_state_dict,
            encoder_prune_attention_layer="encoder.transformer.layers.0.attention.gate_for_layer.gate_threshold" in student_model_state_dict,
            encoder_prune_feed_forward_intermediate="encoder.transformer.layers.0.feed_forward.gate_for_intermediate.gate_threshold" in student_model_state_dict,
            encoder_prune_feed_forward_layer="encoder.transformer.layers.0.feed_forward.gate_for_layer.gate_threshold" in student_model_state_dict,
        )
    )
    model = wav2vec2_model(**config)
    model.load_state_dict(student_model_state_dict, strict=False)
    model = set_tau_min_max(model, eta_min=args.eta_min, eta_max=args.eta_max)
    model = set_tau_reverse(model, reverse_tau=args.reverse_tau)
    model = set_shrink_scale(model, shrink_scale=args.shrink_scale)
    # for learnable tau -> strict=False
    # model.load_state_dict(student_model_state_dict)
    # conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = model.prune(prune_ratio=0.2788) # 94.6M 94570360 960
    # conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = model.prune(prune_ratio=0.2744) # 93685192 100
    conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = model.prune(prune_ratio=args.save_ratio)
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
    print(json.dumps(pruned_config, indent=4))

    ret = {
        "state_dict": model.state_dict(),
        "config": pruned_config,
        "distill_linear_projs": distill_linear_projs_state_dict,
    }
    return ret


def load_pruned_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = wav2vec2_model(**ckpt["config"])
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
        "--save_ratio",
        default=0.5,
        type=float,
        help="save_ratio",
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
    out_path = args.distilled_ckpt.parent / "pruned_hubert_large.pth"
    torch.save(
        prune_from_ckpt(
            distilled_ckpt=args.distilled_ckpt,
            original_ckpt=args.original_ckpt
        ),
        out_path
    )

    # Check if loading from ckpt works
    load_pruned_model(out_path)

    print(f"Successfully saved pruned model weights and config to: {out_path}")
