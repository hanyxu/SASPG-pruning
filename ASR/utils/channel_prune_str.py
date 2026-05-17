"""Structured prune helpers: SASPG (no Gumbel ladder) vs NASP (7-tier Gumbel)."""

from __future__ import annotations

from typing import Any


def saspg_pick_channel_mask(module: Any, keep_ratio: float):
    """Pick one fixed ladder mask from target keep ratio (SASPG str, no NAS over tiers)."""
    tiers = (
        (1.0, module.mask_1),
        (0.75, module.mask_0_75),
        (0.5, module.mask_0_5),
        (0.25, module.mask_0_25),
        (0.125, module.mask_0_125),
    )
    chosen = module.mask_1
    for thr, mask in tiers:
        if keep_ratio <= thr + 1e-5:
            chosen = mask
    return chosen


def set_ffn_pruner_exact_size(module: Any, mask_channel) -> None:
    """Propagate active FFN channel count to weight pruners (SASPG size loss)."""
    n_active = int((mask_channel == 1.0).sum().item())
    for layer in (module.intermediate_dense, module.output_dense):
        qm = layer.weight_pruneizer_saspg.pruner.pruner
        x_shape = getattr(qm, "x_shape", None) or 1
        per_ch = max(1, x_shape // max(1, mask_channel.numel()))
        qm.exact_size = n_active * per_ch


def set_attn_pruner_exact_size(module: Any, mask_channel) -> None:
    n_active = int((mask_channel == 1.0).sum().item())
    for layer in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
        qm = layer.weight_pruneizer_saspg.pruner.pruner
        qm.exact_size = n_active * getattr(module, "head_dim", 64)


def nasp_require_seven_tier_values(value_0_075: float, value_0_1: float) -> None:
    if value_0_075 == 0.0 or value_0_1 == 0.0:
        raise ValueError(
            "NASP str requires non-zero --value-0075 and --value-01 (7-tier Gumbel ladder only)"
        )
