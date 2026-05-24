"""Compatibility shims for newer HuggingFace transformers (test_saspg ships 4.45+)."""

import torch

try:
    from transformers.pytorch_utils import torch_int_div
except ImportError:
    def torch_int_div(tensor, divisor):
        return torch.div(tensor, divisor, rounding_mode="floor")


def force_eager_attn_config(config):
    """Custom PrunedHubert* classes do not support SDPA in transformers 4.45+."""
    if config is None:
        return config
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    return config
