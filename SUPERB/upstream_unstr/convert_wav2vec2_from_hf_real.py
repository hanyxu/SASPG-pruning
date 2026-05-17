"""Convert Hugging Face's Wav2Vec2 to our format."""

import torch
from transformers import Wav2Vec2Model
from torchaudio.models.wav2vec2.utils import import_huggingface_model

from wav2vec2.model import wav2vec2_model


if __name__ == "__main__":
    # 1. 定义输出文件名和 Hugging Face 模型名
    hf_model_name = "/project_bdda8/bdda/hnxu/prune/SSLprune/hf_models/wav2vec2-base"
    
    out_name = "/project_bdda8/bdda/hnxu/prune/SSLprune/DPHuBERT_pretrain_unstr/pretrained/wav2vec2-base.hf.pth"
    
    # 2. 从 Hugging Face 加载原始模型
    original = Wav2Vec2Model.from_pretrained(hf_model_name)
    
    # 3. 使用 torchaudio 工具函数导入模型权重
    #    注意：这里的 import_huggingface_model 是一个通用函数，同样适用于 Wav2Vec2
    imported = import_huggingface_model(original)
    print("Successfully imported model from Hugging Face:")
    print(imported)
    
    # 4. 定义 DPHuBERT 项目所需的 wav2vec2-base 配置字典
    #    这里的很多值与 convert_hubert_from_hf.py 中的 hubert_base_config 保持一致，
    #    以确保实验的统一性。
    wav2vec2_base_config = dict(
        extractor_mode="group_norm",
        extractor_conv_layer_config=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_use_attention=[True] * 12,
        encoder_use_feed_forward=[True] * 12,
        encoder_num_heads=[12] * 12,
        encoder_head_dim=64,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=[3072] * 12,
        encoder_ff_interm_dropout=0.0,  # 与项目中的其他 base 模型保持一致
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,     # wav2vec2 base 使用 post norm
        encoder_layer_drop=0.05,            # 与项目中的其他 base 模型保持一致
        aux_num_out=None,
        normalize_waveform=False,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )

    # 5. 将转换后的模型权重和配置字典保存到 .pth 文件
    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': wav2vec2_base_config,
        }, 
        out_name
    )
    print(f"\nSaved converted model and config to: {out_name}")

    # 6. (可选) 验证保存的检查点是否可以被成功加载
    print("\nVerifying the saved checkpoint...")
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    res = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Verification result -> Missing keys: {res.missing_keys}, Unexpected keys: {res.unexpected_keys}")
    if not res.missing_keys and not res.unexpected_keys:
        print("Verification successful!")