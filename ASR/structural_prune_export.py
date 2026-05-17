# Post-training physical structured prune for channel-pruning runs (SASPG str, NASP str).
# Minimum units: attention heads + FFN intermediate dims per encoder layer only.
# Writes ``<config.base.output_dir>/out/pruned/config.json`` with per-layer
# ``pruned_attention_heads`` / ``pruned_ffn_inter`` plus ``pytorch_model.bin``
# whose weight tensors use **reduced matrix shapes** (true structural shrink),
# loadable by ``models_my.str_modeling_*_minmax_magnitude`` (same contract as prune_ASR_*_mag.py).

import json
import logging
import os
import shutil
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)


def _encoder_layers(model: torch.nn.Module):
    if hasattr(model, "wav2vec2"):
        return model.wav2vec2.encoder.layers, "wav2vec2"
    if hasattr(model, "hubert"):
        return model.hubert.encoder.layers, "hubert"
    return None, None


def _first_eval_row(trainer) -> Optional[Dict[str, Any]]:
    """Trainer.eval_dataset may be a single Dataset or a list/tuple of datasets (multi-eval)."""
    ev = trainer.eval_dataset
    if ev is None:
        return None
    if isinstance(ev, (list, tuple)):
        ds = ev[0]
    else:
        ds = ev
    if ds is None or len(ds) == 0:
        return None
    return ds[0]


def _coerce_input_values_tensor(raw: Union[torch.Tensor, Any], device: torch.device) -> torch.Tensor:
    """Wav2Vec2 expects (batch, time); HF rows may be (time,), (1, time), or over-nested lists."""
    t = torch.as_tensor(raw, dtype=torch.float32, device=device)
    while t.dim() > 2:
        t = t.squeeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def _populate_mask_channel_eval(model: torch.nn.Module, row: Dict[str, Any], device: torch.device) -> None:
    """One forward in eval so each layer picks discrete mask_channel (argmax ladder)."""
    model.eval()
    with torch.no_grad():
        input_values = _coerce_input_values_tensor(row["input_values"], device)
        kwargs = {"input_values": input_values}
        if row.get("attention_mask") is not None:
            am = torch.as_tensor(row["attention_mask"], dtype=torch.long, device=device)
            while am.dim() > 2:
                am = am.squeeze(0)
            if am.dim() == 1:
                am = am.unsqueeze(0)
            kwargs["attention_mask"] = am
        model(**kwargs)


def _copy_tokenizer_sidecars(orig_out_dir: str, pruned_dir: str, vendor: str, release_root: str) -> None:
    if vendor == "wav2vec2":
        pre = os.path.join(release_root, "hf_models", "wav2vec2-base-100h", "preprocessor_config.json")
    else:
        pre = os.path.join(release_root, "hf_models", "hubert-large-ll60k", "preprocessor_config.json")
    if os.path.isfile(pre):
        shutil.copy2(pre, os.path.join(pruned_dir, "preprocessor_config.json"))
    for name in ("vocab.json", "tokenizer_config.json", "special_tokens_map.json"):
        p = os.path.join(orig_out_dir, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(pruned_dir, name))


def export_structural_prune_posttrain(
    q_model: torch.nn.Module,
    hf_config,
    trainer,
    config,
) -> Optional[str]:
    """
    In-place ``prune()`` on each encoder layer, then save dense-pruned checkpoint under ``out/pruned/``.

    Returns pruned_dir path, or None if skipped.
    """
    if not getattr(config.prune_cfg, "channel_pruning", False):
        return None
    reg_type = getattr(config.prune_cfg, "reg_type", "")
    from main_prune import resolve_smoke_reg_type

    reg_impl = resolve_smoke_reg_type(
        reg_type, getattr(getattr(config, "model", None), "model_name", None)
    )
    if reg_impl not in ("channelpruning", "channelpruninghubert"):
        logger.info(
            "structural_prune_export: skip (reg_type=%s -> %s; SASPG/NASP str only)",
            reg_type,
            reg_impl,
        )
        return None

    layers, vendor = _encoder_layers(q_model)
    if layers is None:
        logger.warning("structural_prune_export: skip (no wav2vec2 / hubert backbone on model)")
        return None

    release_root = os.path.dirname(os.path.abspath(__file__))
    orig_out = os.path.join(config.base.output_dir, "out")
    pruned_dir = os.path.join(orig_out, "pruned")
    os.makedirs(pruned_dir, exist_ok=True)

    device = next(q_model.parameters()).device
    row = _first_eval_row(trainer)
    if row is None:
        logger.warning("structural_prune_export: skip (no eval_dataset / empty)")
        return None

    _populate_mask_channel_eval(q_model, row, device)

    config_pruned = deepcopy(hf_config.to_dict())
    pruned_attention_heads = []
    pruned_ffn_inter = []

    q_model.eval()
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
    logger.info("structural_prune_export: wrote %s heads=%s ffn=%s", cfg_path, pruned_attention_heads, pruned_ffn_inter)

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
    filtered = {k: v for k, v in q_model.state_dict().items() if "gate_threshold" not in k}
    pruned_m.load_state_dict(filtered, strict=False)
    bin_path = os.path.join(pruned_dir, "pytorch_model.bin")
    torch.save(pruned_m.state_dict(), bin_path)
    logger.info("structural_prune_export: saved %s", bin_path)

    _copy_tokenizer_sidecars(orig_out, pruned_dir, vendor, release_root)
    return pruned_dir
