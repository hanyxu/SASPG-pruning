#!/usr/bin/env python3
"""One-shot migration: drop smoke-unused modules, rename quant->prune, wanda->legacy keys, main_prune."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- paths to delete (not on smoke critical path) ---
DELETE_PATHS = [
    "models/quantized_wav2vec2_fln_channel_prune_10.py",
    "models/quantized_hubert_fln_channel_prune_10.py",
    "quantization/adaround",
    "quantization/quantizers_direct_backup_20250314.py",
    "quantization/quantizers_direct_orig_NASCP.py",
    "quantization/quantizers_direct_semi.py",
    "quantization/quantizers_direct_taslp_trying_but_abandon_str.py",
    "quantization/quantizers.py",
    "utils/adaround_utils.py",
    "utils/test.py",
    "utils/hf_models_orig.py",
    "utils/compute_macs.py",
    "utils/mac_to_dict.py",
    "pruning_core",
    "run_NASCP_ASR_hubert_minmax_100h_new_value_channel10.sh",
    "run_NASCP_ASR_hubert_minmax_100h_new_value.sh",
    "run_NASCP_ASR_hubert_minmax_960h_new_value_channel10.sh",
    "run_NASCP_ASR_hubert_minmax_960h_new_value.sh",
    "run_NASCP_ASR_w2v_minmax_100h_new_value_channel10.sh",
    "run_NASCP_ASR_w2v_minmax_100h_new_value.sh",
    "run_NASCP_ASR_w2v_minmax_960h_new_value_channel10.sh",
    "run_NASCP_ASR_w2v_minmax_960h_new_value.sh",
    "run_unstr_ASR_hubert_minmax_960h.sh",
    "run_unstr_ASR_w2v_minmax_960h.sh",
]

# --- file renames (relative to ROOT) ---
FILE_RENAMES = [
    ("main_wav_mp.py", "main_prune.py"),
    ("utils/quant_click_options.py", "utils/prune_click_options.py"),
    ("utils/per_embd_quant_utils.py", "utils/per_embd_prune_utils.py"),
    ("quantization/quantizers_direct.py", "pruning/pruners_direct.py"),
    ("quantization/autoquant_utils.py", "pruning/autoprune_utils.py"),
    ("quantization/base_quantized_classes.py", "pruning/base_pruned_classes.py"),
    ("quantization/base_quantized_model.py", "pruning/base_pruned_model.py"),
    ("quantization/quantization_manager_direct.py", "pruning/pruning_manager_direct.py"),
    ("quantization/quant_utils.py", "pruning/prune_utils_module.py"),
    ("models/quantized_wav2vec2_fln_prune.py", "models/pruned_wav2vec2_fln_prune.py"),
    ("models/quantized_wav2vec2_fln_channel_prune.py", "models/pruned_wav2vec2_fln_channel_prune.py"),
    ("models/quantized_wav2vec2_fln_prune_mag.py", "models/pruned_wav2vec2_fln_prune_mag.py"),
    ("models/quantized_hubert_fln_prune.py", "models/pruned_hubert_fln_prune.py"),
    ("models/quantized_hubert_fln_channel_prune.py", "models/pruned_hubert_fln_channel_prune.py"),
    ("models/quantized_hubert_fln_prune_mag.py", "models/pruned_hubert_fln_prune_mag.py"),
]

TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json"}


def delete_unused() -> None:
    for rel in DELETE_PATHS:
        p = ROOT / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            print(f"removed dir: {rel}")
        elif p.is_file():
            p.unlink()
            print(f"removed file: {rel}")


def rename_quantization_dir() -> None:
    qdir = ROOT / "quantization"
    pdir = ROOT / "pruning"
    if qdir.is_dir() and not pdir.exists():
        # move remaining quantization/* into pruning/
        pdir.mkdir(parents=True, exist_ok=True)
        for child in qdir.iterdir():
            dest = pdir / child.name
            if dest.exists():
                if child.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(child), str(dest))
        qdir.rmdir()
        print("moved quantization/ -> pruning/")
    elif qdir.is_dir() and pdir.exists():
        shutil.rmtree(qdir, ignore_errors=True)
        print("dropped leftover quantization/")


def _mv(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if not src.exists():
        if dst.exists():
            return
        print(f"skip missing: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    print(f"renamed: {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def apply_file_renames() -> None:
    rename_quantization_dir()
    for src, dst in FILE_RENAMES:
        sp, dp = ROOT / src, ROOT / dst
        if not sp.exists() and src.startswith("quantization/"):
            sp = ROOT / "pruning" / Path(src).name
        _mv(sp, dp)


def iter_text_files():
    skip = {"smoke_runs_500", "__pycache__", "hf_models", "_temp_weights", "SSLprune_ASR_release_backup"}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in TEXT_SUFFIXES and "backup_" not in str(p):
                yield p


def transform_text(text: str) -> str:
    # Protect HuggingFace / structural flags before global quant->prune
    text = text.replace("config.prune", "@@CFG_PRUNE_BOOL@@")
    text = text.replace("pruned_config.prune", "@@PRUNED_CFG_PRUNE_BOOL@@")
    text = text.replace("config.pruned_", "@@CFG_PRUNED_FIELD@@")
    text = text.replace("pruned_attention_heads", "@@PRUNED_ATTN_HEADS@@")
    text = text.replace("pruned_ffn_inter", "@@PRUNED_FFN_INTER@@")
    text = text.replace("already_pruned", "@@ALREADY_PRUNED@@")
    text = text.replace("max_prune_ratio", "@@MAX_PRUNE_RATIO@@")
    text = text.replace("min_prune_ratio", "@@MIN_PRUNE_RATIO@@")
    text = text.replace("channel_prune10", "@@CHANNEL_PRUNE10@@")
    text = text.replace("channel_pruning", "@@CHANNEL_PRUNING@@")
    text = text.replace("channel_prune", "@@CHANNEL_PRUNE@@")
    text = text.replace("mag_prune", "@@MAG_PRUNE@@")
    text = text.replace("structural_prune", "@@STRUCTURAL_PRUNE@@")
    text = text.replace("prune_only", "@@PRUNE_ONLY@@")
    text = text.replace("pruning_core", "@@PRUNING_CORE@@")
    text = text.replace("pruning_utils", "@@PRUNING_UTILS@@")

    # CLI namespace: config.quant -> config.prune_cfg (avoid clash with config.prune bool)
    text = re.sub(r"\bconfig\.quant\b", "config.prune_cfg", text)

    # wanda legacy keys (before quantizer rename)
    wanda_map = [
        ("wandamaghubert", "mag_mask_hubert"),
        ("wandamag", "mag_mask"),
        ("wandahubert", "saspg_hubert"),
        ("wandalm", "wavlm_mask"),  # unused in smoke but keep distinct
        ("wanda", "saspg"),
        ("weight_quantizer_wanda", "weight_pruner"),
        ("weight_pruner_wanda", "weight_pruner"),
        ("quantizer_wanda", "pruner"),
        ("_wanda", "_mask_pruner"),
    ]
    for old, new in wanda_map:
        text = text.replace(old, new)

    # Longest-first quant variants
    repl = [
        ("QuantizedFromPretrainModel", "PrunedFromPretrainModel"),
        ("QuantizedActivationWrapper", "PrunedActivationWrapper"),
        ("QuantizedActivation", "PrunedActivation"),
        ("QuantizedModule", "PrunedModule"),
        ("QuantizedModel", "PrunedModel"),
        ("QuantizationManager", "PruningManager"),
        ("QuantizationHijacker", "PruningHijacker"),
        ("QuantizerNotInitializedError", "PrunerNotInitializedError"),
        ("AsymmetricUniformQuantizer", "AsymmetricUniformPruner"),
        ("SymmetricUniformQuantizer", "SymmetricUniformPruner"),
        ("quantization_options", "pruning_options"),
        ("activation_quantization_options", "activation_pruning_options"),
        ("transformer_quant_options", "transformer_prune_options"),
        ("train-quantized", "train-pruned"),
        ("no-weight-quant", "no-weight-prune"),
        ("weight_quant", "weight_prune"),
        ("act_quant", "act_prune"),
        ("quantized_", "pruned_"),
        ("Quantized", "Pruned"),
        ("quantizers_direct", "pruners_direct"),
        ("quantizer", "pruner"),
        ("Quantizer", "Pruner"),
        ("quantization", "pruning"),
        ("Quantization", "Pruning"),
        ("quantize_model", "prune_model"),
        ("quantize_", "prune_"),
        ("quantize", "prune"),
        ("quantized", "pruned"),
        ("autoquant", "autoprune"),
        ("AutoQuant", "AutoPrune"),
        ("quant_utils", "prune_utils_module"),
        ("quant_click_options", "prune_click_options"),
        ("make_qparams", "make_prune_params"),
        ("qparams", "prune_params"),
        ("QMethods", "PMethods"),
        ("_quant_a", "_prune_a"),
        ("quant", "prune"),
        ("Quant", "Prune"),
    ]
    for old, new in repl:
        text = text.replace(old, new)

    # Restore protected tokens
    restores = [
        ("@@CFG_PRUNE_BOOL@@", "config.prune"),
        ("@@PRUNED_CFG_PRUNE_BOOL@@", "pruned_config.prune"),
        ("@@CFG_PRUNED_FIELD@@", "config.pruned_"),
        ("@@PRUNED_ATTN_HEADS@@", "pruned_attention_heads"),
        ("@@PRUNED_FFN_INTER@@", "pruned_ffn_inter"),
        ("@@ALREADY_PRUNED@@", "already_pruned"),
        ("@@MAX_PRUNE_RATIO@@", "max_prune_ratio"),
        ("@@MIN_PRUNE_RATIO@@", "min_prune_ratio"),
        ("@@CHANNEL_PRUNE10@@", "channel_prune10"),
        ("@@CHANNEL_PRUNING@@", "channel_pruning"),
        ("@@CHANNEL_PRUNE@@", "channel_prune"),
        ("@@MAG_PRUNE@@", "mag_prune"),
        ("@@STRUCTURAL_PRUNE@@", "structural_prune"),
        ("@@PRUNE_ONLY@@", "prune_only"),
        ("@@PRUNING_CORE@@", "pruning_core"),
        ("@@PRUNING_UTILS@@", "pruning_utils"),
    ]
    for ph, val in restores:
        text = text.replace(ph, val)

    # Fix double-replacements / known artifacts
    fixes = [
        ("main_wav_mp.py", "main_prune.py"),
        ("from pruning.pruning", "from pruning"),
        ("prune_cfg_cfg", "prune_cfg"),
        ("prune_paramsms", "prune_params"),
        ("per_embd_prune_utils", "per_embd_prune_utils"),
        ("hijack_act_prune", "hijack_act_prune"),
        ("hijack_weight_prune", "hijack_weight_prune"),
        ("prepare_model_for_pruneization", "prepare_model_for_pruning"),
        ("set_ffn_pruner_exact_size", "set_ffn_pruner_exact_size"),
        ("set_attn_pruner_exact_size", "set_attn_pruner_exact_size"),
        ("no adaround", "no adaround"),
        ("prune_ASR_", "prune_ASR_"),
        ("models.pruned_", "models.pruned_"),
        ("from models.quantized_", "from models.pruned_"),
        ("from models.pruned_", "from models.pruned_"),
        ("PrunedWav2Vec2ForCTC", "PrunedWav2Vec2ForCTC"),
        ("PrunedHubertForCTC", "PrunedHubertForCTC"),
        ("@@", "@@"),  # noop
    ]
    for old, new in fixes:
        text = text.replace(old, new)

    return text


def patch_main_prune_reg_dispatch(text: str) -> str:
    """Keep only smoke-used reg_type branches in get_*_model loaders."""
    # Already handled by wanda->saspg rename in resolve_smoke_reg_type
    return text


def transform_all_files() -> None:
    for p in iter_text_files():
        if "migrate_quant_to_prune.py" in str(p):
            continue
        if "strip_str_modeling" in str(p):
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        new = transform_text(raw)
        new = patch_main_prune_reg_dispatch(new)
        if new != raw:
            p.write_text(new, encoding="utf-8")
            print(f"patched: {p.relative_to(ROOT)}")


def main() -> int:
    os.chdir(ROOT)
    delete_unused()
    apply_file_renames()
    transform_all_files()
    print("migration done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
