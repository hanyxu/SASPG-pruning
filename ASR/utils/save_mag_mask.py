import torch
import os
from collections import OrderedDict

# --- 1. 配置参数 ---

# 模型权重文件路径（相对本 release 根目录）
_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_BIN_PATH = os.path.join(_RELEASE_ROOT, "hf_models", "wav2vec2-base-100h", "pytorch_model.bin")
# 保存掩码的根目录
BASE_OUTPUT_DIR = "./wav2vec2_base_100h_masks"

# 要计算的保留比例 (10%, 20%, ..., 50%)
# hand_ratio 在您的代码中代表保留比例
THRESHOLD_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]

# 定义需要处理的权重名称的后缀
TARGET_WEIGHT_SUFFIXES = [
    "feed_forward.intermediate_dense.weight",
    "feed_forward.output_dense.weight",
    "attention.q_proj.weight",
    "attention.k_proj.weight",
    "attention.v_proj.weight",
    "attention.out_proj.weight",
]

# --- 2. 实现 create_mask 函数 ---

def create_mask(x: torch.Tensor, threshold_ratio: float) -> torch.Tensor:
    """
    根据张量值的绝对大小创建一个二进制掩码。
    值为 1 的位置对应于绝对值排名前 threshold_ratio 的元素。

    Args:
        x (torch.Tensor): 输入的权重张量。
        threshold_ratio (float): 要保留的权重比例 (例如 0.25 表示保留最大的 25%)。

    Returns:
        torch.Tensor: 与 x 形状相同的二进制掩码张量。
    """
    if x.numel() == 0:
        return torch.empty_like(x)

    # 计算 x 的绝对值
    abs_x = torch.abs(x)
    
    # 展平张量以进行排序
    flat_abs_x = abs_x.flatten()
    
    # 计算阈值的位置 k
    # 使用 round() 确保 k 是整数
    k = round(threshold_ratio * flat_abs_x.numel())
    
    # 如果 k 为 0，则返回全零掩码
    if k == 0:
        return torch.zeros_like(x)
        
    # 找到第 k 大的元素作为阈值
    # 使用 .kthvalue() 更高效
    threshold_value = torch.kthvalue(flat_abs_x, flat_abs_x.numel() - k + 1).values
    
    # 创建掩码：绝对值大于等于阈值的为 1，否则为 0
    # 注意处理所有值都相同的情况，这里 >= 可以确保至少保留 k 个元素
    mask = torch.where(abs_x >= threshold_value, torch.ones_like(x), torch.zeros_like(x))
    
    return mask

# --- 3. 主逻辑 ---

if __name__ == "__main__":
    print(f"正在从 '{MODEL_BIN_PATH}' 加载模型权重...")
    # 加载模型权重字典
    state_dict = torch.load(MODEL_BIN_PATH, map_location='cpu')
    print("模型权重加载完毕。")

    # 遍历所有定义的保留比例
    for ratio in THRESHOLD_RATIOS:
        # 根据比例创建输出文件夹
        ratio_percent = int(ratio * 100)
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"masks_ratio_{ratio_percent}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- 正在为保留比例 {ratio_percent}% 生成掩码 ---")
        print(f"掩码将保存到: {output_dir}")

        # 遍历模型权重字典中的每一项
        for key, weight_tensor in state_dict.items():
            # 检查当前权重是否是我们需要处理的目标
            is_target = any(key.endswith(suffix) for suffix in TARGET_WEIGHT_SUFFIXES)
            
            if is_target:
                # print(f"  正在处理权重: {key}")
                
                # 1. 生成掩码
                mask = create_mask(weight_tensor, ratio)
                
                # 2. 定义保存路径
                # 文件名与权重键名完全相同，以便于后续加载
                output_path = os.path.join(output_dir, key)
                
                # 确保权重键名中的目录结构也存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 3. 保存掩码张量
                torch.save(mask, output_path)

        print(f"保留比例 {ratio_percent}% 的所有目标掩码已生成并保存。")

    print("\n所有任务完成！")
