# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from enum import Enum
from pathlib import Path

from transformers import AutoConfig, HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Processor

from utils.utils import DotDict

logger = logging.getLogger("BENCHMARK")
logger.setLevel(logging.ERROR)

_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_HF_ROOT = _RELEASE_ROOT / "hf_models"


class HF_Models(Enum):
    wav2vec2 = str(_HF_ROOT / "wav2vec2-base")
    wavlm = str(_HF_ROOT / "wavlm-base-plus")
    hubert = str(_HF_ROOT / "hubert-large-ll60k")

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


MODEL_TO_BACKBONE_ATTR = {
    HF_Models.wav2vec2: "wav2vec2",
    HF_Models.hubert: "hubert",
    HF_Models.wavlm: "wavlm",
}


def load_model_and_tokenizer(
    model_name, model_path, use_fast_tokenizer, cache_dir, return_att_mask=False, num_labels=None, **kw
):
    del use_fast_tokenizer, return_att_mask, num_labels, kw

    out = DotDict()
    if model_path is not None:
        model_name_or_path = model_path
    else:
        model_name_or_path = HF_Models[model_name].value

    out.model_name_or_path = model_name_or_path

    if "wav2vec" in model_name:
        config_dir = HF_Models.wav2vec2.value
        config = AutoConfig.from_pretrained(config_dir, cache_dir=cache_dir)
        processor = Wav2Vec2Processor.from_pretrained(config_dir)
        model = Wav2Vec2ForCTC.from_pretrained(model_name_or_path, cache_dir=cache_dir, config=config)
    elif "hubert" in model_name:
        config_dir = HF_Models.hubert.value
        config = AutoConfig.from_pretrained(config_dir, cache_dir=cache_dir)
        processor = Wav2Vec2Processor.from_pretrained(config_dir)
        model = HubertForCTC.from_pretrained(model_name_or_path, cache_dir=cache_dir, config=config)
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
    out.backbone_attr = MODEL_TO_BACKBONE_ATTR.get(out.model_enum, None)
    return out
