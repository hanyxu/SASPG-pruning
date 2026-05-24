#!/usr/bin/env python3
"""
MAG-only ASR: score-based prune export (no CTC training).

Pipeline: baseline FT checkpoint -> out/pruned/ -> single finetune via main_prune (--mag-prune-first).
Does not affect SASPG / NASP training paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

_ASR_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ASR_ROOT not in sys.path:
    sys.path.insert(0, _ASR_ROOT)


def _default_prune_params(
    *,
    max_prune_ratio: float,
    min_prune_ratio: float,
    channel_pruning: bool,
    mag_structural: bool,
    baseline_bin: str,
) -> dict:
    from pruning.pruners_direct import PMethods
    from pruning.range_estimators import OptMethod, RangeEstimators

    return {
        "lmb": 0.0,
        "single_bit": 0,
        "lmb_dis": 0.0,
        "total_steps": 0,
        "weight_prune": False,
        "act_prune": True,
        "method": PMethods.Bayesian_uniform,
        "n_bits": 32,
        "n_bits_act": 32,
        "max_bit": 8,
        "min_bit": 4,
        "value_1": 0.0,
        "value_0_75": 0.0,
        "value_0_5": 0.0,
        "value_0_25": 0.0,
        "value_0_125": 0.0,
        "value_0_1": 0.0,
        "value_0_075": 0.0,
        "max_prune_ratio": max_prune_ratio,
        "min_prune_ratio": min_prune_ratio,
        "fix_prob": False,
        "mag_prune": False,
        "hand_ratio": False,
        "hard": False,
        "only_size_hard": False,
        "decay_tau": False,
        "prune_2_4": False,
        "is_arc_prune": False,
        "per_channel_weights": False,
        "percentile": None,
        "gating_method": "pg",
        "gate_init_dict": {"q2": 0, "q4": 0, "q8": 0.02},
        "include_pruning": False,
        "channel_pruning": channel_pruning,
        "nasp_ladder": False,
        "mag_structural": mag_structural,
        "prune_only": False,
        "reg_type": "mag_str" if mag_structural else "mag_unstr",
        "fixed_bit_dict": {"q8": 0, "q4": 0},
        "prune_setup": "FP_logits",
        "weight_range_method": RangeEstimators.MSE,
        "weight_range_options": {"opt_method": OptMethod.golden_section},
        "eta_max": 0.5,
        "eta_min": 0.01,
        "model_path": baseline_bin,
        "save_path": None,
    }


def _load_baseline(model_name: str, baseline_dir: str):
    from utils.hf_models import load_model_and_tokenizer

    out = load_model_and_tokenizer(
        model_name=model_name,
        model_path=None,
        use_fast_tokenizer=False,
        cache_dir=None,
    )
    if baseline_dir:
        bin_path = os.path.join(baseline_dir, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location="cpu")
            out.model.load_state_dict(state, strict=False)
            logger.info("mag_asr_prune_export: loaded weights from %s", bin_path)
        else:
            logger.warning(
                "mag_asr_prune_export: no pytorch_model.bin under %s, using from_pretrained weights",
                baseline_dir,
            )
    return out.model, out.config


def _wrap_model(org_model, model_name: str, mode: str, prune_ratio: float, baseline_dir: str):
    from main_prune import get_pruned_wav2vec2_model, resolve_smoke_reg_type

    keep_ratio = 1.0 - float(prune_ratio)
    min_ratio = max(0.0, keep_ratio - 0.02)
    baseline_bin = os.path.join(baseline_dir, "pytorch_model.bin")

    if mode == "unstr":
        reg_impl = resolve_smoke_reg_type("mag_unstr", model_name)
        channel_pruning = False
        mag_structural = False
        max_r = prune_ratio
        min_r = max(0.0, prune_ratio - 0.02)
    else:
        reg_impl = resolve_smoke_reg_type("mag_str", model_name)
        channel_pruning = True
        mag_structural = True
        max_r = keep_ratio
        min_r = min_ratio

    PrunedForCTC = get_pruned_wav2vec2_model(reg_impl)
    prune_params = _default_prune_params(
        max_prune_ratio=max_r,
        min_prune_ratio=min_r,
        channel_pruning=channel_pruning,
        mag_structural=mag_structural,
        baseline_bin=baseline_bin,
    )
    return PrunedForCTC(org_model, **prune_params), keep_ratio


def _set_hand_ratio(model, hand_ratio: float):
    for module in model.modules():
        if hasattr(module, "hand_ratio"):
            module.hand_ratio = hand_ratio


def export_mag_unstr(baseline_dir: str, output_dir: str, model_name: str, prune_ratio: float) -> str:
    org_model, hf_config = _load_baseline(model_name, baseline_dir)
    model, _ = _wrap_model(org_model, model_name, "unstr", prune_ratio, baseline_dir)

    model.set_mag_prune()
    _set_hand_ratio(model, float(prune_ratio))

    pruned_dir = os.path.join(output_dir, "out", "pruned")
    os.makedirs(pruned_dir, exist_ok=True)

    from utils.prune_checkpoint_io import collect_prune_sidecar, save_prune_sidecar

    sidecar = collect_prune_sidecar(model)
    if not sidecar:
        raise RuntimeError("mag_asr_prune_export: no magnitude masks collected (unstr)")
    torch.save(model.state_dict(), os.path.join(pruned_dir, "pytorch_model.bin"))
    save_prune_sidecar(pruned_dir, model)
    if hasattr(hf_config, "save_pretrained"):
        hf_config.save_pretrained(pruned_dir)

    from structural_prune_export import _copy_tokenizer_sidecars, _encoder_layers

    _layers, vendor = _encoder_layers(model)
    if vendor:
        _copy_tokenizer_sidecars(baseline_dir, pruned_dir, vendor, _ASR_ROOT)

    logger.info("mag_asr_prune_export: unstr pruned ckpt -> %s (%d sidecar tensors)", pruned_dir, len(sidecar))
    return pruned_dir


def export_mag_str(baseline_dir: str, output_dir: str, model_name: str, prune_ratio: float) -> str:
    from structural_prune_export import (
        _copy_tokenizer_sidecars,
        _encoder_layers,
        _populate_mag_str_masks_from_weights,
    )

    org_model, hf_config = _load_baseline(model_name, baseline_dir)
    model, keep_ratio = _wrap_model(org_model, model_name, "str", prune_ratio, baseline_dir)

    layers, vendor = _encoder_layers(model)
    if layers is None:
        raise RuntimeError("mag_asr_prune_export: no encoder layers on model")

    _populate_mag_str_masks_from_weights(layers)
    logger.info(
        "mag_asr_prune_export: MAG structural masks (keep_ratio=%.4f, target_prune=%.4f)",
        keep_ratio,
        prune_ratio,
    )

    orig_out = os.path.join(output_dir, "out")
    pruned_dir = os.path.join(orig_out, "pruned")
    os.makedirs(pruned_dir, exist_ok=True)

    config_pruned = deepcopy(hf_config.to_dict())
    pruned_attention_heads = []
    pruned_ffn_inter = []

    model.eval()
    with torch.no_grad():
        for layer in layers:
            attn_cfg = layer.attention.prune()
            ffn_cfg = layer.feed_forward.prune()
            pruned_attention_heads.append(int(attn_cfg["num_heads"]))
            pruned_ffn_inter.append(int(ffn_cfg["ff_interm_features"]))

    config_pruned["pruned_attention_heads"] = pruned_attention_heads
    config_pruned["pruned_ffn_inter"] = pruned_ffn_inter
    config_pruned["prune"] = False

    cfg_path = os.path.join(pruned_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_pruned, f, indent=2)

    if vendor == "wav2vec2":
        from models_my.str_modeling_wav2vec2_minmax_magnitude import Wav2Vec2ForCTC as ForCTCOrig
        from transformers import Wav2Vec2Config

        pruned_cfg = Wav2Vec2Config.from_pretrained(pruned_dir)
    else:
        from models_my.str_modeling_hubert_minmax_magnitude import HubertForCTC as ForCTCOrig
        from transformers import HubertConfig

        pruned_cfg = HubertConfig.from_pretrained(pruned_dir)

    pruned_cfg.prune = False
    pruned_m = ForCTCOrig(pruned_cfg)
    filtered = {k: v for k, v in model.state_dict().items() if "gate_threshold" not in k}
    pruned_m.load_state_dict(filtered, strict=False)
    torch.save(pruned_m.state_dict(), os.path.join(pruned_dir, "pytorch_model.bin"))

    release_root = _ASR_ROOT
    _copy_tokenizer_sidecars(orig_out, pruned_dir, vendor, release_root)
    if not os.path.isdir(orig_out):
        os.makedirs(orig_out, exist_ok=True)
    _copy_tokenizer_sidecars(baseline_dir, pruned_dir, vendor, release_root)

    logger.info(
        "mag_asr_prune_export: str pruned ckpt -> %s heads=%s ffn=%s",
        pruned_dir,
        pruned_attention_heads,
        pruned_ffn_inter,
    )
    return pruned_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="MAG ASR prune-only export (no CTC training).")
    parser.add_argument("--mode", choices=("unstr", "str"), required=True)
    parser.add_argument("--model-name", choices=("wav2vec2", "hubert", "w2v"), required=True)
    parser.add_argument("--baseline-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prune-ratio", type=float, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    if model_name == "w2v":
        model_name = "wav2vec2"

    baseline_dir = args.baseline_dir
    if not baseline_dir:
        from utils.hf_models import asr_model_checkpoint

        baseline_dir = asr_model_checkpoint(model_name)

    if args.mode == "unstr":
        export_mag_unstr(baseline_dir, args.output_dir, model_name, args.prune_ratio)
    else:
        export_mag_str(baseline_dir, args.output_dir, model_name, args.prune_ratio)


if __name__ == "__main__":
    main()
