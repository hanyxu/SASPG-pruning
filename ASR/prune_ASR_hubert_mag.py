import json
import pathlib
import torch
import os

from argparse import ArgumentParser
# from models_my.str_modeling_hubert import HubertForCTC as HubertForCTC_orig
# from models_my.str_modeling_hubert_pruned import hubertForCTC as hubertForCTC_pruned
from models_my.str_modeling_hubert_minmax_magnitude import HubertForCTC as HubertForCTC_orig
from copy import deepcopy
from transformers import HubertConfig, AutoConfig
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
        
    for i, layer in enumerate(model.hubert.encoder.layers):
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
    
    pruned_config = HubertConfig.from_pretrained(save_dir)
    pruned_config.prune = False
    # import pdb;pdb.set_trace()
    pruned_model = HubertForCTC_orig(pruned_config)
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
    
    # config = PrunedHubertConfig.from_pretrained(save_dir)
    config = HubertConfig.from_pretrained(save_dir)
    
    # import pdb;pdb.set_trace()
    config.prune = False

    model = HubertForCTC_orig.from_pretrained(
        save_dir,
        config=config,
        # device_map='cpu'
    )
    # import pdb;pdb.set_trace()
    weights_path = f"{save_dir}/pytorch_model.bin"
    model.load_state_dict(torch.load(weights_path))
    # import pdb;pdb.set_trace()
    # model = HubertForCTC_pruned(config)
    # # 加载并过滤权重
    # weights_path = f"{save_dir}/pytorch_model.bin"
    # pretrained_weights = torch.load(weights_path)
    
    # filtered_weights = {
    #     name: param 
    #     for name, param in pretrained_weights.items() 
    #     if 'gate_threshold' not in name
    # }
    
    # model.load_state_dict(filtered_weights, strict=True)
    # model.Hubert.encoder.layers[0].attention.gate_threshold**2
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
    shutil.copy2(os.path.join(root_dir, 'hf_models', 'hubert-large-ll60k', 'preprocessor_config.json'), pruned_dir)  # for hubert

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

    config = AutoConfig.from_pretrained(
        orig_dir,
    )
    config.prune = True

    orig_model = HubertForCTC_orig.from_pretrained(
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
