# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import logging
import os
from enum import Enum
from pathlib import Path

from transformers import AutoConfig, HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Processor

from utils.utils import DotDict

logger = logging.getLogger("BENCHMARK")
logger.setLevel(logging.ERROR)

_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_HF_ROOT = Path(os.environ.get("ASR_HF_MODELS_ROOT", _RELEASE_ROOT / "hf_models"))


class HF_Models(Enum):
    """CLI --model-name choices (paths resolved via asr_model_checkpoint)."""

    wav2vec2 = "wav2vec2"
    wavlm = "wavlm"
    hubert = "hubert"

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


MODEL_TO_BACKBONE_ATTR = {
    "wav2vec2": "wav2vec2",
    "hubert": "hubert",
    "wavlm": "wavlm",
}


def asr_model_checkpoint(model_name: str) -> str:
    """
    ASR fine-tuned checkpoints (CTC + lm_head). Not the SUPERB upstream SSL bundles.

    Defaults (override via env, see validation_smoke_8exp/user_sim/paths.env):
      wav2vec2 -> hf_models/wav2vec2-base-100h
      hubert   -> ASR_HUBERT_MODEL_PATH (user LibriSpeech 100h FT), never hubert-large-ll60k
    """
    name = str(model_name).lower()
    if "wav2vec" in name:
        return os.environ.get(
            "ASR_W2V_MODEL_PATH",
            str(_HF_ROOT / "wav2vec2-base-100h"),
        )
    if "hubert" in name:
        path = os.environ.get("ASR_HUBERT_MODEL_PATH")
        if not path:
            raise ValueError(
                "ASR HuBERT requires ASR_HUBERT_MODEL_PATH pointing to your LibriSpeech 100h "
                "fine-tuned HubertForCTC checkpoint (with lm_head). "
                "Do not use hubert-large-ll60k — that is SUPERB upstream SSL only."
            )
        return path
    if "wavlm" in name:
        raise NotImplementedError("wavlm is not wired in this release build.")
    raise ValueError(f"Unknown model_name: {model_name}")


def is_structural_pruned_checkpoint(checkpoint_dir: str) -> bool:
    """True when ``config.json`` lists per-layer structural head/ffn sizes (MAG/SASPG export)."""
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    heads = cfg.get("pruned_attention_heads")
    return isinstance(heads, list) and len(heads) > 0


def _processor_checkpoint_dir(model_name: str, model_name_or_path: str) -> str:
    """Processor/tokenizer dir: pruned ckpt if complete, else FT baseline."""
    pre = os.path.join(model_name_or_path, "preprocessor_config.json")
    if os.path.isfile(pre):
        return model_name_or_path
    baseline = asr_model_checkpoint(model_name)
    logger.info(
        "ASR processor: using baseline %s (no preprocessor under %s)",
        baseline,
        model_name_or_path,
    )
    return baseline


def load_model_and_tokenizer(
    model_name, model_path, use_fast_tokenizer, cache_dir, return_att_mask=False, num_labels=None, **kw
):
    del use_fast_tokenizer, return_att_mask, num_labels, kw

    out = DotDict()
    if model_path is not None:
        model_name_or_path = model_path
    else:
        model_name_or_path = asr_model_checkpoint(model_name)

    out.model_name_or_path = model_name_or_path
    logger.info("ASR load checkpoint: %s (model_name=%s)", model_name_or_path, model_name)

    processor_dir = _processor_checkpoint_dir(model_name, model_name_or_path)

    if "wav2vec" in model_name:
        if model_path and is_structural_pruned_checkpoint(model_name_or_path):
            from models_my.str_modeling_wav2vec2_minmax_magnitude import (
                Wav2Vec2ForCTC as Wav2Vec2ForCTCStr,
            )
            from transformers import Wav2Vec2Config

            config = Wav2Vec2Config.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            config.prune = False
            processor = Wav2Vec2Processor.from_pretrained(processor_dir)
            model = Wav2Vec2ForCTCStr.from_pretrained(
                model_name_or_path, cache_dir=cache_dir, config=config
            )
        else:
            config_dir = asr_model_checkpoint("wav2vec2")
            config = AutoConfig.from_pretrained(config_dir, cache_dir=cache_dir)
            processor = Wav2Vec2Processor.from_pretrained(processor_dir)
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name_or_path, cache_dir=cache_dir, config=config
            )
    elif "hubert" in model_name:
        if model_path and is_structural_pruned_checkpoint(model_name_or_path):
            from models_my.str_modeling_hubert_minmax_magnitude import HubertForCTC as HubertForCTCStr
            from transformers import HubertConfig

            config = HubertConfig.from_pretrained(
                model_name_or_path, cache_dir=cache_dir, attn_implementation="eager"
            )
            config.prune = False
            processor = Wav2Vec2Processor.from_pretrained(processor_dir)
            model = HubertForCTCStr.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                config=config,
                attn_implementation="eager",
            )
        else:
            config_dir = model_name_or_path
            config = AutoConfig.from_pretrained(
                config_dir, cache_dir=cache_dir, attn_implementation="eager"
            )
            processor = Wav2Vec2Processor.from_pretrained(processor_dir)
            model = HubertForCTC.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                config=config,
                attn_implementation="eager",
            )
    elif "wavlm" in model_name:
        raise NotImplementedError("wavlm is not wired in this release build.")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    out.config = config
    out.model_name_or_path = model_name_or_path
    out.processor = processor
    out.tokenizer = processor.tokenizer
    out.model = model
    out.model_enum = HF_Models[model_name]
    out.backbone_attr = MODEL_TO_BACKBONE_ATTR.get(model_name)
    return out
