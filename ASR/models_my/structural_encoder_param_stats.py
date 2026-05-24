"""Analytic param counts for encoder (attention + FFN) vs structural pruned config.

Matches HF Wav2Vec2 / HuBERT layout: q,k,v are Linear(hidden, h*head_dim), out_proj
Linear(h*head_dim, hidden); FFN is Linear(hidden, f) + Linear(f, hidden).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def count_encoder_layer_attn_ff_params(
    hidden: int,
    heads_nominal: int,
    heads_kept: int,
    ffn_intermediate: int,
) -> int:
    """Return weight+bias element count for one encoder layer (attention + FFN only)."""
    head_dim = hidden // heads_nominal
    qkv_out = heads_kept * head_dim
    attn = 3 * (hidden * qkv_out + qkv_out) + (qkv_out * hidden + hidden)
    ffn = (hidden * ffn_intermediate + ffn_intermediate) + (
        ffn_intermediate * hidden + hidden
    )
    return attn + ffn


def encoder_dense_attn_ff_param_total(config: Any) -> Optional[int]:
    """
    Nominal dense encoder (all layers): attention + FFN weight+bias elements only.

    Uses full `num_attention_heads` and `intermediate_size` (ignores pruned_* lists).
    """
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    heads_nom = int(getattr(config, "num_attention_heads", 0) or 0)
    f_full = int(getattr(config, "intermediate_size", 0) or 0)
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if hidden <= 0 or heads_nom <= 0 or f_full <= 0 or n_layers <= 0:
        return None
    layer = count_encoder_layer_attn_ff_params(
        hidden, heads_nom, heads_nom, f_full
    )
    return int(layer * n_layers)


def encoder_structural_retention_from_config(config: Any) -> Optional[Dict[str, Any]]:
    """
    Compare pruned (per-layer heads / FFN width) vs nominal full-width encoder.

    Returns None if config has no pruned lists (dense baseline only).
    """
    hidden = int(getattr(config, "hidden_size", 0) or 0)
    heads_nom = int(getattr(config, "num_attention_heads", 0) or 0)
    f_full = int(getattr(config, "intermediate_size", 0) or 0)
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if hidden <= 0 or heads_nom <= 0 or f_full <= 0 or n_layers <= 0:
        return None

    ph: Optional[List[int]] = getattr(config, "pruned_attention_heads", None)
    pf: Optional[List[int]] = getattr(config, "pruned_ffn_inter", None)
    if ph is None and pf is None:
        return None

    if ph is None:
        ph = [heads_nom] * n_layers
    if pf is None:
        pf = [f_full] * n_layers
    if len(ph) != n_layers or len(pf) != n_layers:
        return None

    baseline_layer = count_encoder_layer_attn_ff_params(
        hidden, heads_nom, heads_nom, f_full
    )
    baseline_total = baseline_layer * n_layers

    per_layer_ret: List[float] = []
    actual_total = 0
    for i in range(n_layers):
        h_i = int(ph[i])
        f_i = int(pf[i])
        c_i = count_encoder_layer_attn_ff_params(hidden, heads_nom, h_i, f_i)
        actual_total += c_i
        per_layer_ret.append(c_i / baseline_layer if baseline_layer > 0 else 0.0)

    enc_ret = actual_total / baseline_total if baseline_total > 0 else 0.0
    return {
        "baseline_encoder_params": baseline_total,
        "actual_encoder_params": actual_total,
        "encoder_attn_ff_param_retention": enc_ret,
        "encoder_attn_ff_prune_ratio": max(0.0, min(1.0, 1.0 - enc_ret)),
        "per_layer_encoder_param_retention": per_layer_ret,
        "per_layer_encoder_prune_ratio": [
            max(0.0, min(1.0, 1.0 - r)) for r in per_layer_ret
        ],
    }
