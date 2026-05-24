"""Structured prune helpers: SASPG (threshold), NASP (7-tier Gumbel ladder), MAG (top-k)."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Order matches exp_prob / gumble_prob stacking (value-1 .. value-0075).
NASP_TIER_NAMES: Tuple[str, ...] = (
    "1",
    "0_5",
    "0_25",
    "0_125",
    "0_75",
    "0_1",
    "0_075",
)


def normalize_channel_score(sum_score: torch.Tensor) -> torch.Tensor:
    min_val = torch.min(sum_score)
    max_val = torch.max(sum_score)
    return (sum_score - min_val) / (max_val - min_val + 1e-8) + 1e-5


def compute_ffn_channel_score(module: Any) -> torch.Tensor:
    w_in = module.intermediate_dense.weight
    w_out = module.output_dense.weight
    return (w_in**2).sum(dim=1) + (w_out**2).sum(dim=0)


def compute_attn_head_channel_score(module: Any) -> torch.Tensor:
    w_q = module.q_proj.weight
    w_k = module.k_proj.weight
    w_v = module.v_proj.weight
    w_out = module.out_proj.weight
    score = (w_q**2).sum(dim=1) + (w_k**2).sum(dim=1) + (w_v**2).sum(dim=1) + (w_out**2).sum(dim=0)
    return score.reshape(module.num_heads, -1, 1).sum(dim=1)


def saspg_channel_mask_from_threshold(
    module: Any,
    sum_channel_score: torch.Tensor,
    *,
    tau: float,
    fix_prob: bool = False,
    training: bool = True,
    eta_min: float = 0.01,
) -> torch.Tensor:
    prob_score = normalize_channel_score(sum_channel_score)
    threshold_sq = module.threshold_prune_channel

    if not getattr(module, "reset_threshold_channel", False) and not fix_prob:
        with torch.no_grad():
            init_th = torch.sqrt(torch.min(prob_score).clamp(min=1e-7)).to(threshold_sq.device)
            module.threshold_prune_channel.data.copy_(init_th)
            if module.threshold_prune_channel.grad is not None:
                module.threshold_prune_channel.grad.zero_()
        module.reset_threshold_channel = True

    if fix_prob:
        if getattr(module, "get_fix_mask_channel", False) and training and module.mask_channel is not None:
            return module.mask_channel
        mask = (torch.round(torch.sigmoid((prob_score - threshold_sq**2) / eta_min)) == 1.0).to(
            prob_score.dtype
        )
        if training:
            module.get_fix_mask_channel = True
    else:
        mask = torch.sigmoid((prob_score - threshold_sq**2) / (10.0 * tau))
        mask = mask.round() - mask.detach() + mask

    return _squeeze_channel_mask(mask)


def saspg_apply_ffn_channel_mask(module: Any) -> torch.Tensor:
    sum_score = compute_ffn_channel_score(module)
    module.sum_channel_score = sum_score
    module.mask_channel = saspg_channel_mask_from_threshold(
        module,
        sum_score,
        tau=float(module.tau),
        fix_prob=bool(getattr(module, "fix_prob", False)),
        training=module.training,
        eta_min=float(getattr(module, "eta_min", 0.01)),
    )
    set_ffn_pruner_exact_size(module, module.mask_channel)
    return module.mask_channel


def saspg_apply_attn_channel_mask(module: Any) -> torch.Tensor:
    sum_score = compute_attn_head_channel_score(module)
    module.sum_channel_score = sum_score
    module.mask_channel = saspg_channel_mask_from_threshold(
        module,
        sum_score,
        tau=float(module.tau),
        fix_prob=bool(getattr(module, "fix_prob", False)),
        training=module.training,
        eta_min=float(getattr(module, "eta_min", 0.01)),
    )
    set_attn_pruner_exact_size(module, module.mask_channel)
    return module.mask_channel


def nasp_require_seven_tier_values(value_0_075: float, value_0_1: float) -> None:
    if value_0_075 == 0.0 or value_0_1 == 0.0:
        raise ValueError(
            "NASP str requires non-zero --value-0075 and --value-01 (7-tier Gumbel ladder only)"
        )


def nasp_ffn_tier_keep_k(num_channels: int = 4096) -> List[int]:
    """Keep counts for Hubert-large FFN intermediate (4096); matches IS25 NAS-CP."""
    return [num_channels, 3072, 2048, 1024, 512, 409, 307]


def nasp_attn_tier_keep_k(num_heads: int) -> List[int]:
    """Per-tier head keep counts for 7-tier NASP (w2v-base 12h / hubert-large 16h)."""
    if num_heads == 12:
        return [12, 9, 6, 3, 2, 1, 1]
    if num_heads == 16:
        return [16, 12, 8, 4, 2, 2, 1]
    # Generic: fraction of heads per tier name order.
    fracs = (1.0, 0.75, 0.5, 0.25, 0.125, 0.1, 0.075)
    return [max(1, round(num_heads * f)) if f < 1.0 else num_heads for f in fracs]


def nasp_build_tier_masks(
    sum_channel_score: torch.Tensor,
    keep_ks: Sequence[int],
    device: torch.device,
) -> List[torch.Tensor]:
    score = sum_channel_score.squeeze(-1) if sum_channel_score.dim() > 1 else sum_channel_score
    n = score.numel()
    masks: List[torch.Tensor] = []
    for k in keep_ks:
        k_eff = min(max(int(k), 0), n)
        mask = torch.zeros(n, device=device, dtype=score.dtype)
        if k_eff > 0:
            if k_eff >= n:
                mask.fill_(1.0)
            else:
                _, idx = torch.topk(score, k_eff)
                mask[idx] = 1.0
        masks.append(mask.view(-1, 1))
    return masks


def nasp_init_ladder_masks_ffn(module: Any, device: torch.device) -> None:
    sum_score = compute_ffn_channel_score(module)
    module.sum_channel_score = sum_score
    ks = nasp_ffn_tier_keep_k(sum_score.numel())
    masks = nasp_build_tier_masks(sum_score, ks, device)
    (
        module.mask_1,
        module.mask_0_5,
        module.mask_0_25,
        module.mask_0_125,
        module.mask_0_75,
        module.mask_0_1,
        module.mask_0_075,
    ) = masks


def nasp_init_ladder_masks_attn(module: Any, device: torch.device) -> None:
    sum_score = compute_attn_head_channel_score(module)
    module.sum_channel_score = sum_score
    ks = nasp_attn_tier_keep_k(module.num_heads)
    masks = nasp_build_tier_masks(sum_score.squeeze(-1), ks, device)
    (
        module.mask_1,
        module.mask_0_5,
        module.mask_0_25,
        module.mask_0_125,
        module.mask_0_75,
        module.mask_0_1,
        module.mask_0_075,
    ) = masks


def nasp_init_prob_parameters(module: Any) -> None:
    nasp_require_seven_tier_values(module.value_0_075, module.value_0_1)
    if module.value_0_75 == 0.0:
        raise NotImplementedError("NASP ladder requires non-zero --value-075")
    denom = (
        module.value_1
        + module.value_0_5
        + module.value_0_25
        + module.value_0_125
        + module.value_0_75
        + module.value_0_1
        + module.value_0_075
    )
    dev = next(module.parameters()).device
    module.prob_1 = torch.nn.Parameter(
        torch.tensor([module.value_1 / denom], device=dev), requires_grad=True
    )
    module.prob_0_5 = torch.nn.Parameter(
        torch.tensor([module.value_0_5 / denom], device=dev), requires_grad=True
    )
    module.prob_0_25 = torch.nn.Parameter(
        torch.tensor([module.value_0_25 / denom], device=dev), requires_grad=True
    )
    module.prob_0_125 = torch.nn.Parameter(
        torch.tensor([module.value_0_125 / denom], device=dev), requires_grad=True
    )
    module.prob_0_75 = torch.nn.Parameter(
        torch.tensor([module.value_0_75 / denom], device=dev), requires_grad=True
    )
    module.prob_0_1 = torch.nn.Parameter(
        torch.tensor([module.value_0_1 / denom], device=dev), requires_grad=True
    )
    module.prob_0_075 = torch.nn.Parameter(
        torch.tensor([module.value_0_075 / denom], device=dev), requires_grad=True
    )


def nasp_ladder_masks_tuple(module: Any) -> Tuple[torch.Tensor, ...]:
    return (
        module.mask_1,
        module.mask_0_5,
        module.mask_0_25,
        module.mask_0_125,
        module.mask_0_75,
        module.mask_0_1,
        module.mask_0_075,
    )


def nasp_forward_channel_mask(module: Any) -> torch.Tensor:
    """7-tier Gumbel-Softmax mixture (training) or argmax tier (eval)."""
    exp_prob = torch.cat(
        [
            module.prob_1,
            module.prob_0_5,
            module.prob_0_25,
            module.prob_0_125,
            module.prob_0_75,
            module.prob_0_1,
            module.prob_0_075,
        ]
    )
    module.exp_prob = exp_prob
    is_hard = bool(getattr(module, "is_hard", False))
    module.gumble_prob = F.gumbel_softmax(exp_prob, tau=float(module.tau), hard=is_hard)
    ladder = nasp_ladder_masks_tuple(module)
    if module.training:
        mask = sum(p * m for p, m in zip(module.gumble_prob, ladder))
    else:
        idx = int(torch.argmax(exp_prob).item())
        mask = ladder[idx]
    module.mask_channel = mask
    return mask


def mag_num_keep_attn(num_heads: int, all_prune_ratio: float) -> int:
    """Head keep count from target ratio (hubert-large special cases from TASLP mag)."""
    if num_heads == 16:
        table = {0.1: 2, 0.2: 3, 0.3: 5, 0.4: 6, 0.5: 8}
        r = round(all_prune_ratio, 1)
        if r in table:
            return table[r]
    return max(1, round(num_heads * all_prune_ratio))


def mag_num_keep_ffn(num_channels: int, all_prune_ratio: float) -> int:
    return max(1, round(num_channels * all_prune_ratio))


def mag_fixed_channel_mask(
    module: Any,
    sum_channel_score: torch.Tensor,
    num_keep: int,
) -> torch.Tensor:
    score = sum_channel_score.squeeze(-1) if sum_channel_score.dim() > 1 else sum_channel_score
    n = score.numel()
    k = min(max(int(num_keep), 0), n)
    mask = torch.zeros_like(score, dtype=torch.bool)
    if k > 0:
        if k >= n:
            mask[:] = True
        else:
            _, idx = torch.topk(score, k)
            mask[idx] = True
    return mask.view(-1, 1)


def mag_apply_ffn_channel_mask(module: Any) -> torch.Tensor:
    if getattr(module, "mag_pruned", None) is not None and bool(module.mag_pruned.item()):
        return module.mask_channel
    sum_score = compute_ffn_channel_score(module)
    module.sum_channel_score = sum_score
    n_ch = sum_score.numel()
    ratio = float(getattr(module, "all_prune_ratio", 0.5))
    num_keep = mag_num_keep_ffn(n_ch, ratio)
    module.fixed_mask = mag_fixed_channel_mask(module, sum_score, num_keep)
    module.compiled_mask = module.fixed_mask
    module.mask_channel = module.fixed_mask
    if hasattr(module, "mag_pruned"):
        module.mag_pruned.fill_(1.0)
    set_ffn_pruner_exact_size(module, module.mask_channel)
    return module.mask_channel


def mag_apply_attn_channel_mask(module: Any) -> torch.Tensor:
    if getattr(module, "mag_pruned", None) is not None and bool(module.mag_pruned.item()):
        return module.mask_channel
    sum_score = compute_attn_head_channel_score(module)
    module.sum_channel_score = sum_score
    ratio = float(getattr(module, "all_prune_ratio", 0.5))
    num_keep = mag_num_keep_attn(module.num_heads, ratio)
    module.fixed_mask = mag_fixed_channel_mask(module, sum_score.squeeze(-1), num_keep)
    module.compiled_mask = module.fixed_mask
    module.mask_channel = module.fixed_mask.squeeze(-1)
    if hasattr(module, "mag_pruned"):
        module.mag_pruned.fill_(1.0)
    set_attn_pruner_exact_size(module, module.mask_channel)
    return module.mask_channel


def _ffn_channel_unit_param_count(module: Any) -> int:
    """One FFN intermediate unit is [hidden, 1] -> ``hidden`` weights in one Linear row/col."""
    ref = module.intermediate_dense
    qm = ref.weight_pruneizer_saspg.pruner.pruner
    x_shape = int(getattr(qm, "x_shape", 0) or 0)
    n_ch = int(getattr(module, "intermediate_dense", ref).weight.shape[0])
    if x_shape > 0 and n_ch > 0:
        return max(1, x_shape // n_ch)
    return int(getattr(module, "intermediate_dense", ref).weight.shape[1])


def _attn_head_unit_param_count(module: Any) -> int:
    """One attention head unit is [hidden, head_dim] in each of Q/K/V/O (see ``prune()``)."""
    hidden = int(getattr(module, "embed_dim", 0) or 0)
    head_dim = int(getattr(module, "head_dim", 0) or 0)
    if hidden <= 0 or head_dim <= 0:
        num_heads = int(getattr(module, "num_heads", 0) or 0)
        if head_dim <= 0 and num_heads > 0:
            head_dim = max(1, hidden // num_heads) if hidden > 0 else 64
        if hidden <= 0 and head_dim > 0 and num_heads > 0:
            hidden = head_dim * num_heads
    return max(1, hidden * head_dim)


def set_ffn_pruner_exact_size(module: Any, mask_channel) -> None:
    n_active = int(active_channel_mask(mask_channel).sum().item())
    unit = _ffn_channel_unit_param_count(module)
    for layer in (module.intermediate_dense, module.output_dense):
        qm = layer.weight_pruneizer_saspg.pruner.pruner
        qm.exact_size = n_active * unit


def set_attn_pruner_exact_size(module: Any, mask_channel) -> None:
    n_active = int(active_channel_mask(mask_channel).sum().item())
    unit = _attn_head_unit_param_count(module)
    for layer in (module.q_proj, module.k_proj, module.v_proj, module.out_proj):
        qm = layer.weight_pruneizer_saspg.pruner.pruner
        qm.exact_size = n_active * unit


def active_channel_mask(mask_channel: torch.Tensor) -> torch.Tensor:
    if mask_channel.is_floating_point():
        return torch.round(mask_channel) == 1.0
    return mask_channel == 1.0


def _squeeze_channel_mask(mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() > 2:
        mask = mask.squeeze(-1)
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    return mask


def flatten_channel_mask(mask_channel: torch.Tensor) -> torch.Tensor:
    m = active_channel_mask(mask_channel)
    while m.dim() > 1:
        m = m.squeeze(-1)
    return m.reshape(-1)


def kept_channel_indices(mask_channel: torch.Tensor) -> torch.Tensor:
    flat = flatten_channel_mask(mask_channel)
    return flat.nonzero(as_tuple=True)[0].long()


def count_kept_channels(mask_channel: torch.Tensor) -> int:
    return int(flatten_channel_mask(mask_channel).sum().item())


def ensure_export_mask_channel(module: Any) -> torch.Tensor:
    if getattr(module, "mask_channel", None) is not None:
        return module.mask_channel
    if getattr(module, "nasp_ladder", False):
        raise RuntimeError("NASP export: run a forward pass before prune() to set mask_channel")
    if getattr(module, "mag_structural", False):
        if hasattr(module, "intermediate_dense"):
            return mag_apply_ffn_channel_mask(module)
        return mag_apply_attn_channel_mask(module)
    if hasattr(module, "intermediate_dense"):
        return saspg_apply_ffn_channel_mask(module)
    return saspg_apply_attn_channel_mask(module)
