"""Persist / restore unstructured prune state (SASPG thresholds, magnitude masks)."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PRUNE_SIDECAR_NAME = "prune_sidecar.pt"


def _leaf_pruner(module: nn.Module):
    """SyBayesianBitsPruner inside weight_pruneizer_saspg."""
    wp = getattr(module, "weight_pruneizer_saspg", None)
    if wp is None:
        return None
    pruner = getattr(wp, "pruner", None)
    if pruner is None:
        return None
    return getattr(pruner, "pruner", None)


def iter_weight_prune_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module, object]]:
    for name, mod in model.named_modules():
        leaf = _leaf_pruner(mod)
        if leaf is not None and hasattr(leaf, "threshold_prune"):
            yield name, mod, leaf


def _mask_tensor(leaf, mod: nn.Module) -> Optional[torch.Tensor]:
    if getattr(leaf, "mag_mask", None) is not None:
        return leaf.mag_mask
    if getattr(leaf, "mask", None) is not None:
        return leaf.mask
    weight = getattr(mod, "weight", None)
    hand_ratio = getattr(leaf, "hand_ratio", None)
    if weight is not None and hand_ratio not in (None, False) and float(hand_ratio) > 0:
        if hasattr(leaf, "create_mask"):
            return leaf.create_mask(weight.data, float(hand_ratio))
    return None


def collect_prune_sidecar(model: nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, mod, leaf in iter_weight_prune_modules(model):
        if hasattr(leaf, "threshold_prune"):
            out[f"{name}.threshold_prune"] = leaf.threshold_prune.detach().cpu()
        mask = _mask_tensor(leaf, mod)
        if mask is not None:
            out[f"{name}.mag_mask"] = mask.detach().cpu()
    return out


def save_prune_sidecar(checkpoint_dir: str, model: nn.Module) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, PRUNE_SIDECAR_NAME)
    payload = collect_prune_sidecar(model)
    torch.save(payload, path)
    logger.info("prune_checkpoint_io: saved %d tensors to %s", len(payload), path)
    return path


def load_prune_sidecar(checkpoint_dir: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    path = os.path.join(checkpoint_dir, PRUNE_SIDECAR_NAME)
    if not os.path.isfile(path):
        return {}
    return torch.load(path, map_location=map_location)


def apply_prune_sidecar(model: nn.Module, sidecar: Dict[str, torch.Tensor], device=None) -> int:
    """Apply sidecar tensors to live pruners. Returns number of tensors applied."""
    if not sidecar:
        return 0
    by_prefix: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in sidecar.items():
        if key.endswith(".threshold_prune"):
            prefix = key[: -len(".threshold_prune")]
            by_prefix.setdefault(prefix, {})["threshold_prune"] = tensor
        elif key.endswith(".mag_mask"):
            prefix = key[: -len(".mag_mask")]
            by_prefix.setdefault(prefix, {})["mag_mask"] = tensor

    applied = 0
    for name, mod, leaf in iter_weight_prune_modules(model):
        bundle = by_prefix.get(name)
        if not bundle:
            continue
        if "threshold_prune" in bundle:
            t = bundle["threshold_prune"]
            if device is not None:
                t = t.to(device)
            leaf.threshold_prune.data.copy_(t.reshape_as(leaf.threshold_prune.data))
            applied += 1
        if "mag_mask" in bundle:
            m = bundle["mag_mask"]
            if device is not None:
                m = m.to(device)
            leaf.mag_mask = m
            leaf.mask = m
            applied += 1
    logger.info("prune_checkpoint_io: applied %d sidecar entries", applied)
    return applied


def checkpoint_dir_from_model_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    if os.path.isdir(model_path):
        return model_path
    return os.path.dirname(model_path)


def load_prune_sidecar_into_model(model: nn.Module, model_path: Optional[str]) -> int:
    ckpt_dir = checkpoint_dir_from_model_path(model_path)
    if not ckpt_dir:
        return 0
    sidecar = load_prune_sidecar(ckpt_dir)
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else None
    return apply_prune_sidecar(model, sidecar, device=device)
