import json
import pathlib
import torch
import os

from argparse import ArgumentParser
from models_my.str_modeling_wav2vec2_minmax_magnitude import Wav2Vec2ForCTC as Wav2Vec2ForCTC_orig
# from models_my.str_modeling_wav2vec2_pruned import Wav2Vec2ForCTC as Wav2Vec2ForCTC_pruned

from copy import deepcopy
from transformers import Wav2Vec2Config, AutoConfig
import shutil
import sys
# 2025.5.23 11:23 modified

def set_tau_min_max(model, eta_min=0.01, eta_max=0.5):
    def apply_tau_min_max(module):
        if hasattr(module, 'eta_min'):
            module.eta_min = eta_min
            module.eta_max = eta_max
    
    for name, module in model.named_modules():
        # if isinstance(module, Gate):
        apply_tau_min_max(module)
    
    return model

# class PrunedWav2Vec2Config(Wav2Vec2Config):
#     def __init__(self, pruned_attention_heads=None, pruned_ffn_inter=None, **kwargs):
#         super().__init__(**kwargs)
#         self.pruned_attention_heads = pruned_attention_heads or {}
#         self.pruned_ffn_inter = pruned_ffn_inter or {}
    
def prune_and_save_model(model, config, save_dir):
    
    config_pruned = deepcopy(config.to_dict())
    
    pruned_attention_heads = []
    pruned_ffn_inter = []
    
    model = set_tau_min_max(model, eta_min=args.eta_min, eta_max=args.eta_max)
        
    for i, layer in enumerate(model.wav2vec2.encoder.layers):
        attention = layer.attention
        attn_config = attention.prune()
        
        feed_forward = layer.feed_forward
        ffn_config = feed_forward.prune()
        
        pruned_attention_heads.append(attn_config["num_heads"])
        pruned_ffn_inter.append(ffn_config["ff_interm_features"])
        
    config_pruned['pruned_attention_heads'] = pruned_attention_heads
    config_pruned['pruned_ffn_inter'] = pruned_ffn_inter
    
    # config_dict = config.to_dict()
    print("pruned_config: ")
    import pprint
    pprint.pprint(config_pruned)

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config_pruned, f, indent=2)  # indent=2 使 JSON 易读

    print(f"config saved to {save_dir}/config.json")

    # config.save_pretrained(save_dir)
    # model.save_pretrained(save_dir)
    
    pruned_config = Wav2Vec2Config.from_pretrained(save_dir)
    pruned_config.prune = False
    # import pdb;pdb.set_trace()
    pruned_model = Wav2Vec2ForCTC_orig(pruned_config)
    # 加载并过滤权重
    # weights_path = f"{save_dir}/pytorch_model.bin"
    # pretrained_weights = torch.load(weights_path)
    # for name, param in model.state_dict().items():
    #     if 'gate_threshold' in name:
    #         print(param**2)
            
    filtered_weights = {
        name: param 
        for name, param in model.state_dict().items() 
        if 'gate_threshold' not in name
    }
    
    pruned_model.load_state_dict(filtered_weights, strict=False)
    
    torch.save(pruned_model.state_dict(), f"{save_dir}/pytorch_model.bin")

    print('save_dir:', save_dir)
    
    
    return config

def load_pruned_model(save_dir):
    
    # config = PrunedWav2Vec2Config.from_pretrained(save_dir)
    config = Wav2Vec2Config.from_pretrained(save_dir)
    
    # import pdb;pdb.set_trace()
    config.prune = False

    model = Wav2Vec2ForCTC_orig.from_pretrained(
        save_dir,
        config=config,
        # device_map='cpu'
    )
    # import pdb;pdb.set_trace()
    weights_path = f"{save_dir}/pytorch_model.bin"
    model.load_state_dict(torch.load(weights_path))
    # import pdb;pdb.set_trace()
    # model = Wav2Vec2ForCTC_pruned(config)
    # # 加载并过滤权重
    # weights_path = f"{save_dir}/pytorch_model.bin"
    # pretrained_weights = torch.load(weights_path)
    
    # filtered_weights = {
    #     name: param 
    #     for name, param in pretrained_weights.items() 
    #     if 'gate_threshold' not in name
    # }
    
    # model.load_state_dict(filtered_weights, strict=True)
    # model.wav2vec2.encoder.layers[0].attention.gate_threshold**2
    # import pdb;pdb.set_trace()
    total_params = 0
    trainable_params = 0  
    
    for param in model.parameters():
        # 
        param_count = param.numel()  # 
        total_params += param_count
        
        # 
        if param.requires_grad:
            trainable_params += param_count
    
    # 打印统计结果
    print(f"prund model total param: {total_params:,}")  # 
    print(f"prund model trainable param: {trainable_params:,}")
    print(f"prund model untrainable param: {total_params - trainable_params:,}")
    
    params_dir_path = os.path.join(save_dir, str(total_params))
    os.makedirs(params_dir_path, exist_ok=True)
    print(f"Created new directory for parameter count: {params_dir_path}")
    
    return model

def copy_files_except_specific(orig_dir, pruned_dir, exclude_files):
   
    if not os.path.exists(pruned_dir):
        os.makedirs(pruned_dir)
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(os.path.join(root_dir, 'hf_models', 'wav2vec2-base-100h', 'preprocessor_config.json'), pruned_dir)  # for w2v

    for item in os.listdir(orig_dir):
        src_path = os.path.join(orig_dir, item)
        
        if not os.path.isfile(src_path):
            continue

        if item in exclude_files:
            continue
            
        dest_path = os.path.join(pruned_dir, item)
        
        try:
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)
                print(f"copy: {src_path} -> {dest_path}")
            elif os.path.isdir(src_path):
                # 递归复制目录
                shutil.copytree(src_path, dest_path)
                print(f"copy: {src_path} -> {dest_path}")
        except Exception as e:
            print(f"copy failed: {src_path} -> {dest_path}, error: {e}")


def parse_args():
    parser = ArgumentParser(description="Prune and save distilled model.")
    parser.add_argument(
        "--distilled_ckpt",
        type=pathlib.Path,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--orig_dir",
        type=pathlib.Path,
        help="Path to the distilled model checkpoint."
    )
    parser.add_argument(
        "--pruned_dir",
        type=str,
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
    
    orig_dir = args.orig_dir

    pruned_dir = str(args.orig_dir) + '/pruned/'
    # os.makedirs(pruned_dir, exist_ok=True)

    exclude_files = ['pytorch_model.bin', 'config.json', 'optimizer.pt', 'rng_state.pth', 'scheduler.pt', 'trainer_state.json', 'training_args.bin']
    
    copy_files_except_specific(orig_dir, pruned_dir, exclude_files)    

    orig_dir = os.path.abspath(str(orig_dir))
    cfg_path = os.path.join(orig_dir, "config.json")
    if os.path.isfile(cfg_path):
        config = AutoConfig.from_pretrained(orig_dir)
    else:
        # PrunedWav2Vec2ForCTC is nn.Module-only: Trainer checkpoints omit config.json.
        # Reconstruct from release bundle + checkpoint vocab (for ASR vocab_size).
        root_dir = os.path.dirname(os.path.abspath(__file__))
        bundle = os.path.join(root_dir, "hf_models", "wav2vec2-base-100h")
        bundle_cfg = os.path.join(bundle, "config.json")
        if not os.path.isfile(bundle_cfg):
            raise OSError(
                f"Can't load config for {orig_dir!r} (no config.json in checkpoint) and "
                f"missing bundle config {bundle_cfg}"
            )
        config = AutoConfig.from_pretrained(bundle)
        vocab_path = os.path.join(orig_dir, "vocab.json")
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            if vocab:
                config.vocab_size = max(int(t) for t in vocab.values()) + 1
    config.prune = True

    orig_model = Wav2Vec2ForCTC_orig.from_pretrained(
        orig_dir,config=config)
        # low_cpu_mem_usage=False, 
        # torch_dtype=torch.float32, 
        # device_map='cuda' if torch.cuda.is_available() else 'cpu' )

    total_params = 0
    trainable_params = 0  
    
    for param in orig_model.parameters():
        # 
        param_count = param.numel()  # 
        total_params += param_count
        
        # 
        if param.requires_grad:
            trainable_params += param_count
    
    # 打印统计结果
    print(f"orig total param: {total_params:,}")  # 
    print(f"orig trainable param: {trainable_params:,}")
    print(f"orig untrainable param: {total_params - trainable_params:,}")
    
    orig_model = orig_model.to('cpu')
    orig_model.eval()
    prune_and_save_model(orig_model, config, pruned_dir)
    
    load_pruned_model(pruned_dir)
    
    print(f"Successfully saved pruned model weights and config to:from {orig_dir} to {pruned_dir}")
