"""Prune-ratio helpers for ASR SASPG / NASP str training (not retention)."""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Tuple, Union

import torch

# Log message suffix for encoder_attn_ff_prune_ratio (same key, different counting).
PRUNE_RATIO_LOG_NOTE = {
    "str_channel": (
        "1-exact_size/sum_shape, str channel: exact_size from head [hidden,head_dim] "
        "and FFN [hidden,1] units via mask_channel; sum_shape=nominal encoder attn+FF weight+bias"
    ),
    "mag_str_channel": (
        "1-exact_size/sum_shape, MAG str train (dense shape): channel units from mag score mask; "
        "post-export inference uses physical param ratio on shape-reduced ckpt"
    ),
    "mag_unstr": (
        "1-exact_size/sum_shape, MAG unstr train (dense shape): mag_mask element count; "
        "stage2 uses fixed mag_mask×weight; shape-reduced export is separate"
    ),
    "mag_str_physical": (
        "encoder_attn_ff_prune_ratio from pruned_attention_heads/pruned_ffn_inter vs "
        "nominal dense encoder param count (shape already reduced)"
    ),
    "unstr": (
        "1-exact_size/sum_shape, unstr: exact_size=sum of encoder attn+FF weight elements "
        "with SASPG gate mask==1; sum_shape=nominal encoder attn+FF weight element count"
    ),
}


def encoder_attn_ff_retention(exact_size: float, sum_shape: float) -> float:
    """Fraction of nominal encoder attn+FF params still active (exact_size/sum_shape)."""
    s = float(sum_shape)
    if s <= 0:
        return 0.0
    return max(0.0, min(1.0, float(exact_size) / s))


def encoder_attn_ff_prune_ratio(exact_size: float, sum_shape: float) -> float:
    """Encoder attn+FF prune ratio = 1 - retention."""
    return max(0.0, min(1.0, 1.0 - encoder_attn_ff_retention(exact_size, sum_shape)))


def _encoder_layers(backbone: Any):
    """Resolve encoder layer list from ForCTC wrapper or Pruned*Model backbone."""
    if hasattr(backbone, "wav2vec2"):
        backbone = backbone.wav2vec2
    elif hasattr(backbone, "hubert"):
        backbone = backbone.hubert
    enc = getattr(backbone, "encoder", None)
    if enc is not None and hasattr(enc, "layers"):
        return enc.layers
    raise AttributeError(
        "backbone must be PrunedWav2Vec2Model, PrunedHubertModel, or *ForCTC with "
        ".wav2vec2 / .hubert child"
    )


def iter_encoder_attn_ff_pruning_managers(backbone: Any) -> Iterator[Any]:
    """PruningManagers on encoder Q/K/V/O + FFN Linears only (matches sum_shape scope)."""
    for layer in _encoder_layers(backbone):
        for linear in (
            layer.attention.q_proj,
            layer.attention.k_proj,
            layer.attention.v_proj,
            layer.attention.out_proj,
            layer.feed_forward.intermediate_dense,
            layer.feed_forward.output_dense,
        ):
            pm = getattr(linear, "weight_pruneizer_saspg", None)
            if pm is not None and hasattr(pm, "get_exact_size_prune"):
                yield pm


def encoder_attn_ff_exact_size_from_backbone(backbone: Any) -> float:
    """Sum exact_size over encoder attn+FF pruners (unstr: mask==1 weight count per pruner)."""
    return sum(float(pm.get_exact_size_prune()) for pm in iter_encoder_attn_ff_pruning_managers(backbone))


def encoder_attn_ff_gate_loss_prune_from_backbone(backbone: Any) -> float:
    """Size regularizer numerator; equals encoder_attn_ff_exact_size for unstr."""
    return sum(float(pm.get_gate_loss_prune()) for pm in iter_encoder_attn_ff_pruning_managers(backbone))


def encoder_attn_ff_gate_loss_prune_channel_from_backbone(backbone: Any) -> float:
    """Channel-str size loss: sum regularizer_size_prune_channel over encoder attn+FF only."""
    return sum(
        float(pm.get_gate_loss_prune_channel())
        for pm in iter_encoder_attn_ff_pruning_managers(backbone)
    )


def param_prune_ratio(current: float, dense: float) -> float:
    """Prune ratio from current vs dense parameter counts."""
    d = float(dense)
    if d <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(current) / d))


def minmax_gate_prune_loss(
    ctc_q_loss: Any,
    size_loss: Any,
    lmb: Any,
    exact_prune_ratio: float,
    min_prune_ratio: float,
    max_prune_ratio: float,
) -> Tuple[Any, str]:
    """
    Three-branch min-max schedule on **prune ratio** (higher = more pruned).

    - exact < min: under-pruned -> ctc + lmb * size_loss
    - exact > max: over-pruned -> ctc - lmb * size_loss
    - min <= exact <= max: on target -> ctc only
    """
    if exact_prune_ratio < min_prune_ratio:
        loss = ctc_q_loss + lmb * size_loss
        formula = (
            "formula_1: exact_prune_ratio < min_prune_ratio (under-pruned) "
            "=> loss = ctc_q_loss + lmb * size_loss"
        )
    elif exact_prune_ratio > max_prune_ratio:
        loss = ctc_q_loss - lmb * size_loss
        formula = (
            "formula_2: exact_prune_ratio > max_prune_ratio (over-pruned) "
            "=> loss = ctc_q_loss - lmb * size_loss"
        )
    elif min_prune_ratio <= exact_prune_ratio <= max_prune_ratio:
        loss = ctc_q_loss
        formula = (
            "formula_3: min_prune_ratio <= exact_prune_ratio <= max_prune_ratio "
            "=> loss = ctc_q_loss"
        )
    else:
        raise NotImplementedError(
            f"unhandled prune ratio {exact_prune_ratio} vs "
            f"[{min_prune_ratio}, {max_prune_ratio}]"
        )
    return loss, formula


def loss_breakdown_scalar(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().cpu().item())
    return float(x)


def log_pruned_ctc_loss_breakdown(
    logger: logging.Logger,
    model_tag: str,
    *,
    prune_kind: str,
    ctc_q_loss: Any,
    lmb: Any,
    size_loss: Any,
    total_loss: Any,
    exact_size: float,
    sum_shape: float,
    exact_prune_ratio: float,
    min_prune_ratio: float,
    max_prune_ratio: float,
    loss_formula: Optional[str],
    phase: str = "train",
) -> None:
    """Training/eval debug: exact_prune_ratio and encoder_attn_ff_prune_ratio are the same value."""
    ratio_note = PRUNE_RATIO_LOG_NOTE.get(
        prune_kind, PRUNE_RATIO_LOG_NOTE["unstr"]
    )
    pr = float(exact_prune_ratio)
    sz_term = (
        lmb * size_loss
        if isinstance(lmb, torch.Tensor) or isinstance(size_loss, torch.Tensor)
        else float(lmb) * float(size_loss)
    )
    if isinstance(sz_term, torch.Tensor):
        sz_scalar = loss_breakdown_scalar(sz_term)
    else:
        sz_scalar = float(sz_term)
    logger.info(
        "%s loss breakdown (%s): ctc_q_loss=%s lmb=%s size_loss=%s lmb*size_loss=%s "
        "total_loss=%s exact_size=%s sum_shape=%s "
        "exact_prune_ratio=%s encoder_attn_ff_prune_ratio=%s "
        "(%s; both names are 1-exact_size/sum_shape) "
        "target_prune_ratio_band=[min=%s, max=%s] loss_formula=%s",
        model_tag,
        phase,
        loss_breakdown_scalar(ctc_q_loss),
        loss_breakdown_scalar(lmb),
        loss_breakdown_scalar(size_loss),
        sz_scalar,
        loss_breakdown_scalar(total_loss),
        float(exact_size),
        float(sum_shape),
        pr,
        pr,
        ratio_note,
        float(min_prune_ratio),
        float(max_prune_ratio),
        loss_formula or "n/a",
    )


def prune_kind_from_reg_type(reg_type: str) -> str:
    """Map config prune_cfg.reg_type (incl. smoke aliases) to log prune_kind."""
    r = (reg_type or "").lower()
    if any(
        x in r
        for x in (
            "str",
            "channel",
            "nasp",
            "saspg_hubert_str",
            "saspg_str",
            "mag_str",
            "nasp_str",
        )
    ):
        return "str_channel"
    return "unstr"


def resolve_for_ctc_encoder_backbone(for_ctc_model: Any) -> Tuple[Optional[Any], Optional[float]]:
    """Return (wav2vec2|hubert backbone, sum_shape) for Pruned*ForCTC."""
    sum_shape = getattr(for_ctc_model, "sum_shape", None)
    if sum_shape is None:
        return None, None
    backbone = getattr(for_ctc_model, "wav2vec2", None) or getattr(for_ctc_model, "hubert", None)
    if backbone is None:
        return None, None
    return backbone, float(sum_shape)


def materialize_encoder_masks_one_forward(for_ctc_model: Any, input_values: Any) -> None:
    """One eval forward (no labels) so encoder pruners refresh exact_size from eval masks."""
    for_ctc_model.eval()
    dev = next(for_ctc_model.parameters()).device
    iv = input_values
    if not isinstance(iv, torch.Tensor):
        iv = torch.tensor(iv)
    iv = iv.to(dev)
    if iv.dim() == 1:
        iv = iv.unsqueeze(0)
    with torch.no_grad():
        for_ctc_model(input_values=iv)


def log_structural_encoder_prune_ratio_before_inference(
    for_ctc_model: Any,
    logger: logging.Logger,
    *,
    phase: str = "inference_pre_map",
) -> Optional[float]:
    """MAG/SASPG str after export: ratio from reduced config (physical encoder params)."""
    cfg = getattr(for_ctc_model, "config", None)
    if cfg is None:
        return None
    try:
        from models_my.structural_encoder_param_stats import (
            encoder_structural_retention_from_config,
        )
    except ImportError:
        return None
    stats = encoder_structural_retention_from_config(cfg)
    if stats is None:
        return None
    pr = float(stats["encoder_attn_ff_prune_ratio"])
    logger.info(
        "%s encoder attn+FF prune summary (%s): encoder_attn_ff_prune_ratio=%s "
        "actual_encoder_params=%s baseline_encoder_params=%s (%s)",
        type(for_ctc_model).__name__,
        phase,
        pr,
        int(stats["actual_encoder_params"]),
        int(stats["baseline_encoder_params"]),
        PRUNE_RATIO_LOG_NOTE["mag_str_physical"],
    )
    return pr


def log_encoder_attn_ff_prune_ratio_before_inference(
    for_ctc_model: Any,
    logger: logging.Logger,
    *,
    prune_kind: str = "unstr",
    eval_dataset: Any = None,
    phase: str = "inference_pre_map",
) -> Optional[float]:
    """
    Log encoder_attn_ff_prune_ratio once before WER dataset.map (SASPG / MAG dense-shape models).

    WER map calls model(input_values) without labels, so training loss breakdown is skipped.
    For shape-reduced str checkpoints (HubertForCTC / Wav2Vec2ForCTC with pruned_* config),
    use log_encoder_prune_ratio_before_inference() instead.
    """
    backbone, sum_shape = resolve_for_ctc_encoder_backbone(for_ctc_model)
    if backbone is None or sum_shape is None:
        logger.warning(
            "skip encoder_attn_ff_prune_ratio pre-map log: missing backbone or sum_shape on %s",
            type(for_ctc_model).__name__,
        )
        return None

    for_ctc_model.eval()
    input_values = None
    if eval_dataset is not None:
        try:
            n = len(eval_dataset)
        except TypeError:
            n = 0
        if n > 0:
            sample = eval_dataset[0]
            input_values = sample.get("input_values") if isinstance(sample, dict) else None

    if input_values is not None:
        materialize_encoder_masks_one_forward(for_ctc_model, input_values)
        exact_size_a = encoder_attn_ff_exact_size_from_backbone(backbone)
        materialize_encoder_masks_one_forward(for_ctc_model, input_values)
        exact_size_b = encoder_attn_ff_exact_size_from_backbone(backbone)
        if abs(exact_size_a - exact_size_b) > 1.0:
            logger.warning(
                "encoder_attn_ff exact_size differs across two eval forwards: %s vs %s",
                exact_size_a,
                exact_size_b,
            )
        exact_size = exact_size_b
    else:
        logger.warning(
            "encoder_attn_ff pre-map: no eval sample; using pruner exact_size from last forward"
        )
        exact_size = encoder_attn_ff_exact_size_from_backbone(backbone)

    pr = encoder_attn_ff_prune_ratio(exact_size, sum_shape)
    if prune_kind == "str_channel":
        mask_note = (
            "eval: str channel masks from frozen SASPG/NASP gates; "
            "encoder_attn_ff_prune_ratio constant during dataset.map"
        )
    else:
        mask_note = (
            "eval: unstr hard binary_mask from frozen weight+threshold_prune "
            "(deterministic per forward); encoder_attn_ff_prune_ratio constant during dataset.map"
        )

    logger.info(
        "%s encoder attn+FF prune summary (%s): exact_size=%s sum_shape=%s "
        "exact_prune_ratio=%s encoder_attn_ff_prune_ratio=%s (%s) "
        "masks_fixed_in_eval=True; %s",
        type(for_ctc_model).__name__,
        phase,
        float(exact_size),
        float(sum_shape),
        pr,
        pr,
        PRUNE_RATIO_LOG_NOTE.get(prune_kind, PRUNE_RATIO_LOG_NOTE["unstr"]),
        mask_note,
    )
    return pr


def log_encoder_prune_ratio_before_inference(
    for_ctc_model: Any,
    logger: logging.Logger,
    *,
    prune_kind: str = "unstr",
    reg_type: str = "",
    eval_dataset: Any = None,
    phase: str = "inference_pre_map",
) -> Optional[float]:
    """
    Dispatch pre-map encoder prune-ratio logging by pipeline.

    - Exported str (``pruned_attention_heads`` / ``pruned_ffn_inter``): physical param ratio.
    - Pruned*ForCTC with ``sum_shape`` (SASPG / MAG unstr / MAG str train): gate or mag_mask counts.
    """
    pr = log_structural_encoder_prune_ratio_before_inference(
        for_ctc_model, logger, phase=phase
    )
    if pr is not None:
        return pr
    r = (reg_type or "").lower()
    if "mag_unstr" in r or prune_kind == "unstr" and "mag" in r:
        pk = "mag_unstr"
    elif "mag_str" in r or prune_kind == "str_channel" and "mag" in r:
        pk = "mag_str_channel"
    else:
        pk = prune_kind
    return log_encoder_attn_ff_prune_ratio_before_inference(
        for_ctc_model,
        logger,
        prune_kind=pk,
        eval_dataset=eval_dataset,
        phase=phase,
    )


def format_encoder_attn_ff_prune_ratio_message(
    model_tag: str,
    exact_size: float,
    sum_shape: float,
    *,
    extra: str = "",
) -> Tuple[float, str]:
    """Build a log line using prune ratio (not retention) for encoder attn+FF."""
    pr = encoder_attn_ff_prune_ratio(exact_size, sum_shape)
    msg = (
        f"{model_tag} encoder_attn_ff_prune_ratio={pr} "
        f"(exact_size={float(exact_size)} sum_shape={float(sum_shape)}; 1-exact_size/sum_shape)"
    )
    if extra:
        msg = f"{msg}; {extra}"
    return pr, msg
