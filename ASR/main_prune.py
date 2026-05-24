#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import gc
import logging
import os
import shutil

import re
from functools import partial
from time import time
import warnings
from datasets import load_dataset
try:
    from datasets import load_metric
except ImportError:
    # Newer datasets removes load_metric; keep backward compatibility.
    from evaluate import load as load_metric


def _load_wer_metric():
    """Load WER metric without interactive trust_remote_code prompt (batch-safe)."""
    try:
        return load_metric("wer", trust_remote_code=True)
    except TypeError:
        return load_metric("wer")

from copy import deepcopy
from functools import partial
from pathlib import Path
from pprint import pformat

import click
import numpy as np
import torch
import math
import json

from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator, EarlyStoppingCallback

# Smoke grid aliases: METHOD_MODE (see smoke_experiment_matrix.md) -> legacy module keys.
_SMOKE_REG_ALIASES = (
    "saspg_unstr",
    "saspg_str",
    "mag_unstr",
    "mag_str",
    "nasp_str",
)


def resolve_smoke_reg_type(reg_type: str, model_name: str = None) -> str:
    """Resolve clear smoke --reg-type names to legacy implementation keys."""
    if reg_type not in _SMOKE_REG_ALIASES:
        return reg_type
    hubert = model_name is not None and "hubert" in str(model_name).lower()
    if reg_type == "saspg_unstr":
        return "saspg_hubert" if hubert else "saspg"
    if reg_type == "saspg_str":
        return "channelpruninghubert" if hubert else "channelpruning"
    if reg_type == "mag_unstr":
        return "mag_mask_hubert" if hubert else "mag_mask"
    if reg_type == "mag_str":
        return "channelpruninghubert" if hubert else "channelpruning"
    if reg_type == "nasp_str":
        return "channelpruninghubert" if hubert else "channelpruning"
    return reg_type


def get_pruned_wav2vec2_model(source):
    """Smoke-release model dispatch: SASPG / magnitude / NASP str paths only."""
    if source == 'saspg':
        from models.pruned_wav2vec2_fln_prune import PrunedWav2Vec2ForCTC
    elif source == 'mag_mask':
        from models.pruned_wav2vec2_fln_prune_mag import PrunedWav2Vec2ForCTC
    elif source == 'channelpruning':
        from models.pruned_wav2vec2_fln_channel_prune import PrunedWav2Vec2ForCTC
    elif source == 'saspg_hubert':
        from models.pruned_hubert_fln_prune import PrunedHubertForCTC as PrunedWav2Vec2ForCTC
    elif source == 'mag_mask_hubert':
        from models.pruned_hubert_fln_prune_mag import PrunedHubertForCTC as PrunedWav2Vec2ForCTC
    elif source == 'channelpruninghubert':
        from models.pruned_hubert_fln_channel_prune import PrunedHubertForCTC as PrunedWav2Vec2ForCTC
    else:
        raise ValueError(f'Invalid or unsupported reg_type implementation: {source!r}')
    return PrunedWav2Vec2ForCTC


from torch.nn import MSELoss
from utils import (
    # click options
    pruning_options,
    # activation_pruning_options,
    # qat_options,
    # adaround_options,
    make_prune_params,
    benchmark_options,
    transformer_base_options,
    transformer_data_options,
    transformer_model_options,
    transformer_training_options,
    transformer_progress_options,
    transformer_prune_options,

    # pruning
    prepare_model_for_pruning,
    pass_data_for_range_estimation,
    hijack_act_prune,
    hijack_weight_prune,
    hijack_act_prune_modules,
    hijack_weight_prune_modules,

    # pipeline
    load_model_and_tokenizer,
    load_task_data,
    BENCHMARK_Task,

    # misc
    DotDict,
    Stopwatch,
    
    #macs
    # get_macs,
    # return_dict
)
from utils.prune_ratio_utils import (
    log_encoder_prune_ratio_before_inference,
    prune_kind_from_reg_type,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def _smoke_eval_test_only() -> bool:
    """When set, LibriSpeech inference WER only on test-clean + test-other (skip dev/val_other)."""
    return os.environ.get("SMOKE_EVAL_TEST_ONLY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

# setup logger
########################################################
########################################################
########################################################
# torch.manual_seed(1000) # set seed
########################################################
########################################################
########################################################
logger = logging.getLogger('main')
logger.setLevel(os.environ.get('LOGLEVEL', 'INFO'))


def _log_encoder_prune_ratio_before_wer_map(config, model, dataset, phase: str) -> None:
    """One-shot encoder prune ratio before WER dataset.map (SASPG gates or MAG/str physical)."""
    reg_type = getattr(getattr(config, "prune_cfg", None), "reg_type", "") or ""
    prune_kind = prune_kind_from_reg_type(resolve_smoke_reg_type(str(reg_type)))
    log_encoder_prune_ratio_before_inference(
        model,
        logger,
        prune_kind=prune_kind,
        reg_type=str(reg_type),
        eval_dataset=dataset,
        phase=phase,
    )
# seed=1000
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

# setup stuff
class Config(DotDict):
    pass

pass_config = click.make_pass_decorator(Config, ensure=True)

@click.group()
def benchmark():
    # import pdb;pdb.set_trace()
    logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))

# show default values for all options
click.option = partial(click.option, show_default=True)


from torch.optim import AdamW

def create_custom_optimizer(model, lr_lambda, weight_decay):
    # 分离参数组
    lambda_params = []
    other_params = []
    
    # 递归遍历所有参数
    for name, param in model.named_parameters():
        if 'lambda' in name.lower():  # 匹配所有包含gate的参数
            lambda_params.append(param)
            print(f"Detected gate parameter: {name}")
        else:
            other_params.append(param)

    # 定义不同参数组的学习率策略
    # 这里给gate参数10倍学习率，其他保持默认
    params = [
        {"params": lambda_params, "lr": lr_lambda},
        {"params": other_params}
    ]
    
    return AdamW(params, lr=learning_rate, weight_decay=weight_decay)

chars_to_ignore_regex = '[\,\?\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]) + " "
    return batch

def _is_non_empty_dir(path):
    return path.exists() and len(list(path.iterdir()))


def _decode_export_dir(config):
    override = os.environ.get("SSLPRUNE_DECODE_DIR")
    if override:
        return override
    base = config.base.output_dir or "."
    return os.path.join(base, "decode")


def _decode_run_subdir(config):
    p = Path(config.base.output_dir or "run")
    return "_".join(p.parts[-4:] if len(p.parts) >= 4 else p.parts) + "_last_"

def map_to_result(batch, model, processor, return_att_mask=False):
    # model.precision_levels = [2]
    model.eval()     # do not need 
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        if return_att_mask:
            attention_mask = torch.tensor(batch["attention_mask"], device="cuda").unsqueeze(0)
            logits = model(input_values, attention_mask=attention_mask).logits
        else:
            logits = model(input_values)['logits']

    logits = logits[0] if isinstance(logits, tuple) else logits # depth:8
    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    return batch
        
def _dataloader_num_workers_from_env() -> int:
    """HF DataLoader workers; high counts on NFS often hang after dataset load. Override with DATALOADER_NUM_WORKERS."""
    raw = os.environ.get("DATALOADER_NUM_WORKERS", "32")
    try:
        n = int(raw)
    except ValueError:
        n = 32
    return max(0, min(n, 128))


def _filtered_train_cache_valid(cache_path: str) -> bool:
    """True only if save_to_disk finished (dataset_dict.json or state.json present)."""
    return os.path.isfile(os.path.join(cache_path, "dataset_dict.json")) or os.path.isfile(
        os.path.join(cache_path, "state.json")
    )


def _save_filtered_train_cache_enabled() -> bool:
    return os.environ.get("SAVE_FILTERED_TRAIN_CACHE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dataloader_pin_memory_from_env(num_workers: int) -> bool:
    if num_workers <= 0:
        return False
    v = os.environ.get("DATALOADER_PIN_MEMORY", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def _training_save_strategy(config):
    """HF save_strategy: MAG finetune-after-prune must save checkpoints even when eval is off."""
    eval_strategy = config.progress.eval_strategy
    reg = getattr(getattr(config, "prune_cfg", None), "reg_type", "") or ""
    if (
        reg in _SMOKE_REG_ALIASES
        and str(reg).startswith("mag_")
        and getattr(config.prune_cfg, "mag_prune_first", False)
        and config.progress.save_steps
        and getattr(config.progress, "save_model", False)
    ):
        return "steps"
    return eval_strategy


def _make_huggingface_training_args(config):
    """Create Training Arguments as required by HuggingFace Trainer."""
    output_dir = config.base.output_dir
    if output_dir is not None:
        output_dir = os.path.join(output_dir, 'out')
    _nw = _dataloader_num_workers_from_env()
    _save_strategy = _training_save_strategy(config)
    _eval_strategy = config.progress.eval_strategy
    _load_best = config.progress.load_best_model_at_end
    if _load_best and _save_strategy != _eval_strategy:
        logger.info(
            "TrainingArguments: disabling load_best_model_at_end "
            "(save_strategy=%s != evaluation_strategy=%s)",
            _save_strategy,
            _eval_strategy,
        )
        _load_best = False
    _save_safetensors = True
    if "prune_cfg" in config:
        # Pruned SASPG/MAG models alias threshold tensors; safetensors save fails.
        _save_safetensors = False
    args = TrainingArguments(
        # fp16=True, # 7.28 1:30 修正
        length_column_name="input_length",
        group_by_length=True,
        dataloader_drop_last=False,
        # ignore_data_skip=True, # default set to False
        output_dir=output_dir,
        # overwrite_output_dir=config.base.overwrite_output,
        seed=config.base.seed,
        dataloader_num_workers=_nw,
        dataloader_pin_memory=_dataloader_pin_memory_from_env(_nw),
        do_train=config.training.do_train,
        do_eval=config.training.do_eval,
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        learning_rate=config.training.learning_rate,
        # weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        num_train_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        # warmup_steps=config.training.warmup_steps,
        warmup_ratio=0.1,
        disable_tqdm=not config.progress.tqdm,
        evaluation_strategy=_eval_strategy,
        eval_steps=config.progress.eval_steps, # if type(config.progress.eval_steps)==int else float(config.progress.eval_steps),
        save_strategy=_save_strategy,
        logging_first_step=config.progress.logging_first_step,
        logging_steps=config.progress.logging_steps,
        save_steps=config.progress.save_steps, #  if type(config.progress.save_steps)==int else float(config.progress.save_steps),
        save_total_limit=config.progress.save_total_limit,
        run_name=config.progress.run_name,
        load_best_model_at_end=_load_best,
        metric_for_best_model=config.progress.metric_for_best_model,
        greater_is_better=config.progress.greater_is_better,
        save_safetensors=_save_safetensors,
        # lr_scheduler_type='cosine',
    )
    # import pdb;pdb.set_trace()
    # total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    return args

def _load_and_filter_datasets(config, task_data, processor, return_att_mask=False):
    from datasets import Audio
    from utils.librispeech_data import dataset_cache_dir

    def prepare_dataset(batch):
        audio = batch["file"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        if return_att_mask:
            batch["attention_mask"] = processor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            ).attention_mask[0]
        batch["input_length"] = len(batch["input_values"])
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    datasets = task_data.datasets
    datasets = datasets.map(remove_special_characters)
    datasets = datasets.cast_column("file", Audio(sampling_rate=16000))
    cache_dir = str(dataset_cache_dir())
    os.makedirs(cache_dir, exist_ok=True)

    if config.data.type == "960":
        cache_file = os.path.join(cache_dir, "processed_datasets_960.arrow")
    elif config.data.type == "100":
        cache_file = os.path.join(cache_dir, "processed_datasets_100.arrow")
    else:
        cache_file = os.path.join(cache_dir, f"processed_datasets_{config.data.type}.arrow")

    if os.path.exists(cache_file):
        logger.info(
            "Loading preprocessed dataset from %s (load_from_disk may take a long time on slow storage)",
            cache_file,
        )
        t0 = time()
        datasets = datasets.load_from_disk(cache_file)
        logger.info(
            f"Loaded cached dataset in {time() - t0:.1f}s, splits={list(datasets.keys())}"
        )
    else:
        logger.info(f"Processing dataset and saving to cache at {cache_file}")
        datasets = datasets.map(
            prepare_dataset,
            remove_columns=datasets.column_names["train"],
            num_proc=32,
            load_from_cache_file=True,
        )
        datasets.save_to_disk(cache_file)

    if "wav2vec" in str(config.model.model_name):
        if config.data.type == "100":
            max_input_length_in_sec = 19.2
        if config.data.type == "960":
            max_input_length_in_sec = 19.1
    elif "hubert" in str(config.model.model_name):
        if config.data.type == "100":
            max_input_length_in_sec = 19.2
        elif config.data.type == "960":
            max_input_length_in_sec = 25.0

    _max_len_key = str(max_input_length_in_sec).replace(".", "p")
    filtered_train_cache = os.path.join(
        cache_dir, f"train_filtered_{config.data.type}_{_max_len_key}"
    )
    if os.path.exists(filtered_train_cache) and not _filtered_train_cache_valid(
        filtered_train_cache
    ):
        logger.warning(
            f"Removing incomplete filtered train cache (e.g. disk quota): {filtered_train_cache}"
        )
        shutil.rmtree(filtered_train_cache, ignore_errors=True)

    if _save_filtered_train_cache_enabled() and _filtered_train_cache_valid(
        filtered_train_cache
    ):
        logger.info(f"Loading filtered train cache from {filtered_train_cache}")
        datasets["train"] = datasets.load_from_disk(filtered_train_cache)
    else:
        _n_train = len(datasets["train"])
        logger.info(
            f"Filtering train (max {max_input_length_in_sec}s, examples={_n_train}) ..."
        )
        _filter_np = int(os.environ.get("DATASET_FILTER_NUM_PROC", "8"))
        _filter_kw = {}
        if _filter_np > 0:
            _filter_kw["num_proc"] = min(_filter_np, 32)
        t0 = time()
        datasets["train"] = datasets["train"].filter(
            lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate,
            input_columns=["input_length"],
            **_filter_kw,
        )
        logger.info(
            f"Filtered train size={len(datasets['train'])} in {time() - t0:.1f}s"
        )
        if _save_filtered_train_cache_enabled():
            logger.info(f"Saving filtered train cache to {filtered_train_cache}")
            try:
                datasets["train"].save_to_disk(filtered_train_cache)
            except OSError as exc:
                logger.warning(
                    f"Could not save filtered train cache ({exc}); continuing without cache."
                )
                shutil.rmtree(filtered_train_cache, ignore_errors=True)
    return datasets


def _make_datasets_and_trainer(
    config,
    model,
    model_enum,
    tokenizer,
    task_data,
    processor,
    training_args,
    return_att_mask=False,
    padding=None,
    teacher_model=None,
    datasets_prepared=None,
):
    if datasets_prepared is not None:
        logger.info("Reusing in-memory datasets (skip NFS load_from_disk and train filter)")
        datasets = datasets_prepared
    else:
        datasets = _load_and_filter_datasets(
            config, task_data, processor, return_att_mask=return_att_mask
        )

    from transformers import Wav2Vec2Processor

    # max_input_length_in_sec = 2
    # datasets["validation"] = datasets["validation"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
    # import pdb;pdb.set_trace()
    if 'train' in datasets.keys():
        train_dataset = datasets['train']
    if 'validation' in datasets.keys():
        eval_dataset = datasets["validation"]
    if 'val_other' in datasets.keys():
        eval_other_dataset = datasets["val_other"]
    if 'test_other' in datasets.keys():
        test_other_dataset = datasets["test_other"]
    
    wer_metric = _load_wer_metric()
    logger.info('WER_Metric:')    
    
    def compute_metrics(pred):
        # import pdb;pdb.set_trace()
        pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        """pred.predictions[1] is label !"""
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_logits[pred_logits == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_logits)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
        
        return {"wer": wer}

    def preprocess_logits_for_metrics(logits, labels): # 3-depth
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        # import pdb;pdb.set_trace()
        logits1= logits
        pred_ids1 = torch.argmax(logits1, dim=-1)

        return pred_ids1, labels
        
        
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional, Union
    
    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
        """

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    return_tensors="pt",
                )
            
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch
        
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    from torch.optim.lr_scheduler import LambdaLR

    
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
                outputs = model(**inputs)
                # Compute and split your losses
                total_loss = outputs['loss']
                # if 'loss0' in outputs:
                #     loss0 = outputs['loss0']
                # else:
                #     loss0 = 0
                if 'loss1' in outputs:
                    loss1 = outputs['loss1']
                else:
                    loss1 = 0
                if 'loss2' in outputs:
                    loss2 = outputs['loss2']
                else:
                    loss2 = 0
                if 'loss3' in outputs:
                    loss3 = outputs['loss3']
                else:
                    loss3 = 0
                if 'loss4' in outputs:
                    loss4 = outputs['loss4']
                else:
                    loss4 = 0
                if 'loss5' in outputs:
                    loss5 = outputs['loss5']
                else:
                    loss5 = 0
                if 'loss6' in outputs:
                    loss6 = outputs['loss6']
                else:
                    loss6 = 0
                if 'loss7' in outputs:
                    loss7 = outputs['loss7']
                else:
                    loss7 = 0
                if 'loss8' in outputs:
                    loss8 = outputs['loss8']
                else:
                    loss8 = 0


                if 'kl_div' in outputs:
                    kl_div = outputs['kl_div']
                else:
                    kl_div = 0

                if 'kl_div_1' in outputs:
                    kl_div_1 = outputs['kl_div_1']
                else:
                    kl_div_1 = 0
                
                if 'kl_div_2' in outputs:
                    kl_div_2 = outputs['kl_div_2']
                else:
                    kl_div_2 = 0
                
                if 'kl_div_3' in outputs:
                    kl_div_3 = outputs['kl_div_3']
                else:
                    kl_div_3 = 0
                
                if 'kl_div_4' in outputs:
                    kl_div_4 = outputs['kl_div_4']
                else:
                    kl_div_4 = 0

                if 'kl_div_5' in outputs:
                    kl_div_5 = outputs['kl_div_5']
                else:
                    kl_div_5 = 0

                if 'kl_div_6' in outputs:
                    kl_div_6 = outputs['kl_div_6']
                else:
                    kl_div_6 = 0

                # Return the total loss and other losses
                if return_outputs:
                    return (
                        total_loss,
                        loss1,
                        loss2,
                        loss3,
                        loss4,
                        loss5,
                        loss6,
                        loss7,
                        loss8,
                        kl_div,
                        kl_div_1,
                        kl_div_2,
                        kl_div_3,
                        kl_div_4,
                        kl_div_5,
                        kl_div_6,
                        outputs,
                    )
                return total_loss

        def _save(self, output_dir=None, state_dict=None, **kwargs):
            """Persist HF config.json + full pruned state_dict + prune_sidecar (masks/thresholds)."""
            if not output_dir:
                return
            try:
                super()._save(output_dir=output_dir, state_dict=state_dict, **kwargs)
            except TypeError:
                super()._save(output_dir, state_dict)
            except RuntimeError as exc:
                if "share memory" not in str(exc).lower():
                    raise
                logger.warning(
                    "CustomTrainer: safetensors save failed (shared tensors); using pytorch_model.bin only"
                )
            try:
                full_sd = self.model.state_dict()
                torch.save(full_sd, os.path.join(output_dir, "pytorch_model.bin"))
                from utils.prune_checkpoint_io import save_prune_sidecar

                save_prune_sidecar(output_dir, self.model)
            except Exception:
                logger.exception("CustomTrainer: failed to save full pruned checkpoint in %s", output_dir)
            cfg_path = os.path.join(output_dir, "config.json")
            if os.path.isfile(cfg_path):
                return
            mod = self.model
            cfg = getattr(mod, "config", None)
            if cfg is not None and hasattr(cfg, "save_pretrained"):
                try:
                    cfg.save_pretrained(output_dir)
                    logger.info("CustomTrainer: wrote missing config.json under %s", output_dir)
                except Exception:
                    logger.warning(
                        "CustomTrainer: could not save config.json to %s", output_dir, exc_info=True
                    )

    trainer = CustomTrainer(
        model=model, # the model already pruned here
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        # eval_dataset=[eval_dataset, eval_other_dataset, test_other_dataset],
        eval_dataset=[eval_dataset.select(range(1)), eval_other_dataset.select(range(1)), test_other_dataset.select(range(1))],
        tokenizer=tokenizer,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=20)],
    )
    return trainer, datasets, train_dataset, eval_dataset, processor


def _mag_finetune_skips_prune_wrapper(config) -> bool:
    """MAG str finetune: structural ``out/pruned/`` is already shape-reduced; no Pruned* wrap."""
    if not getattr(getattr(config, "prune_cfg", None), "mag_prune_first", False):
        return False
    mp = getattr(getattr(config, "model", None), "model_path", None)
    if not isinstance(mp, str) or not os.path.isdir(mp):
        return False
    from utils.hf_models import is_structural_pruned_checkpoint

    return is_structural_pruned_checkpoint(mp)


def _prune_model(config, model, model_enum):
    if _mag_finetune_skips_prune_wrapper(config):
        logger.info(
            "mag_prune_first: skip Pruned wrapper; finetune structural ckpt at %s",
            config.model.model_path,
        )
        return model

    _smoke_alias = config.prune_cfg.reg_type
    _reg_impl = resolve_smoke_reg_type(
        _smoke_alias, getattr(config.model, "model_name", None)
    )
    config.prune_cfg.nasp_ladder = _smoke_alias == "nasp_str"
    config.prune_cfg.mag_structural = _smoke_alias == "mag_str"
    PrunedWav2Vec2ForCTC = get_pruned_wav2vec2_model(_reg_impl)
    # suffixes = ['attention', 'out', 'KL', 'MSE','mse', 'FP', '248', 'step2', 'ctc4', 'ctc5', 'noearly', 'thre', 'gumble', '5ctc', 'scale', 'no2', 'base', '111','no4','freeze','fp8','NAS','cnnq','all']
    # if any(config.prune_cfg.reg_type.endswith(suffix) for suffix in suffixes):
    #     config.prune_cfg.reg_type = 'distilctc'
    prune_params = make_prune_params(config)
    prune_params["nasp_ladder"] = getattr(config.prune_cfg, "nasp_ladder", False)
    prune_params["mag_structural"] = getattr(config.prune_cfg, "mag_structural", False)
    # --value-* Gumbel ladder weights: NASP str only (see run_smoke_nasp).
    if _smoke_alias != "nasp_str":
        for _vk in (
            "value_1", "value_0_75", "value_0_5", "value_0_25",
            "value_0_125", "value_0_1", "value_0_075",
        ):
            if _vk in prune_params:
                prune_params[_vk] = 0.0
    if prune_params.get("nasp_ladder"):
        from utils.channel_prune_str import nasp_require_seven_tier_values

        nasp_require_seven_tier_values(
            float(prune_params.get("value_0_075", 0)),
            float(prune_params.get("value_0_1", 0)),
        )
    if isinstance(config.model.model_path, str):
        prune_params['model_path'] = config.model.model_path + '/pytorch_model.bin'
    if isinstance(config.model.thre_model_path, str):
        prune_params['thre_model_path'] = config.model.thre_model_path + '/pytorch_model.bin'
    prune_params['save_path'] = config.base.output_dir # config.model.model_path
    # 将prune_params = make_prune_params(config)保存到config.base.output_dir中
    # import pdb;pdb.set_trace()
    # from copy import deepcopy
    # prune_params_cp = deepcopy(prune_params)
    # prune_params_cp.pop('weight_range_options', None)
    # prune_params_cp.pop('method', None)
    # prune_params_cp.pop('weight_range_method', None)
    # prune_params_save_path = os.path.join(config.base.output_dir, 'prune_params.json')
    
    # with open(prune_params_save_path, 'w') as f:
    #     json.dump(prune_params_cp, f, indent=4)
    
    model = PrunedWav2Vec2ForCTC(model, **prune_params)

    if isinstance(config.model.model_path, str):
        from utils.prune_checkpoint_io import load_prune_sidecar_into_model

        load_prune_sidecar_into_model(model, prune_params.get("model_path"))
    
    # use double precision if necessary
    assert not config.double
    if config.double:
        for m in model.modules():
            if hasattr(m, 'weight') or hasattr(m, 'bias'):
                m.double()

    # set state
    # model.set_prune_state(weight_prune=config.prune_cfg.weight_prune, act_prune=config.prune_cfg.act_prune)

    # print model
    # logger.info('Pruned model:')
    # logger.info(model)

    return model

def _prepare_pruned_model(config, model, loader):
    """Prepare pruned model for training/validation."""
    if config.training.do_train:
        # import pdb;pdb.set_trace()
        model = prepare_model_for_pruning(config, model, loader)
        # 先pass 再开启forward中的可选参数？
    return model

def set_minmax_prune_ratio(model, max_prune_ratio=0.5, min_prune_ratio=0.0):
    def apply_prune_ratio(module):
        if hasattr(module, 'max_prune_ratio'):
            module.max_prune_ratio = max_prune_ratio
        if hasattr(module, 'min_prune_ratio'):
            module.min_prune_ratio = min_prune_ratio
    for name, module in model.named_modules():
        apply_prune_ratio(module)
    
    return model

def set_total_steps(model, total_steps=None):
    def apply_total_steps(module):
        if hasattr(module, 'total_steps'):
            module.total_steps = total_steps
    
    for name, module in model.named_modules():
        apply_total_steps(module)
    
    return model


def set_warmup_ratio(model, warmup_ratio=0.1):
    for module in model.modules():
        if hasattr(module, 'warmup_ratio'):
            module.warmup_ratio = warmup_ratio
    return model

def set_hand_ratio(model, hand_ratio=None):
    def apply_hand_ratio(module):
        if hasattr(module, 'hand_ratio'):
            module.hand_ratio = hand_ratio
    
    for name, module in model.named_modules():
        apply_hand_ratio(module)
    
    return model

def set_tau_min_max(model, eta_min=0.01, eta_max=0.5):
    def apply_tau_min_max(module):
        if hasattr(module, 'eta_min'):
            module.eta_min = eta_min
            module.eta_max = eta_max
    
    for name, module in model.named_modules():
        apply_tau_min_max(module)
    
    return model

# def set_prune_trans(model):
#     def apply_prune_trans(module):
#         if hasattr(module, 'prune_heads'):
#             module.prune_heads = True
#         if hasattr(module, 'prune_intermediate'):
#             module.prune_intermediate = True
    
#     for name, module in model.named_modules():
#         apply_prune_trans(module)
    
#     return model

def _run_task(config, task_data, model_data):
    """Common routine to run training/validation on a signle task."""
    # import pdb;pdb.set_trace()
    model = model_data.model
    model_enum = model_data.model_enum
    tokenizer = model_data.tokenizer
    processor = model_data.processor

    if config.training.do_train:
        # create dirpath if not exist
        os.makedirs(config.base.output_dir, exist_ok=True)

        # log config additionaly into a separate file
        output_dir = config.base.output_dir
        config_file = os.path.join(output_dir, 'config.out')

        if os.path.exists(config_file):
            # 文件已存在，创建一个新的文件名
            new_config_file = os.path.join(output_dir, 'config_1.out')
            with open(new_config_file, 'w') as f:
                f.write(pformat(config) + '\n')
        else:
            # 文件不存在，直接写入
            with open(config_file, 'w') as f:
                f.write(pformat(config) + '\n')

    # prepare training arguments for huggingface Trainer
    training_args = _make_huggingface_training_args(config)
        
    logger.info(f'Training/evaluation parameters for Trainer: {training_args}')

    # attach layer number
    backbone_attr = model_data.backbone_attr

    if backbone_attr is None:
        raise NotImplementedError(
            f'Model {config.model.model_name} not yet supported for ' f'TensorBoard visualization.'
        )
    layers = getattr(model, backbone_attr).encoder.layers
        
    num_layers = len(layers)
    for layer_idx, layer in enumerate(layers):
        for m in layer.modules():
            m.layer_idx = layer_idx
            m.num_layers = num_layers

    # Pruning!
    if 'prune_cfg' in config:
        # replace model with a pruned one
        model = _prune_model(config, model, model_enum)

        
    # Per-embedding / per-token pruning
    # per_token = config.get('prune', {}).get('per_token', False)
    # per_embd = config.get('prune', {}).get('per_embd', False)
    # per_groups = config.get('prune', {}).get('per_groups', None)
    # permute = config.get('prune', {}).get('per_groups_permute', False)
    # base_axis = 2 if (per_embd or per_groups) else 1
    
    
    # Mixed-precision control for weight. pruners


    datasets_shared = None
    _mag_struct_finetune = _mag_finetune_skips_prune_wrapper(config)

    # Prepare pruned model for training/validation
    if "prune_cfg" in config and not _mag_struct_finetune:
        # pass
        # make another trainer with individually controlled padding strategy & batch size for
        # range estimation
        config_ = deepcopy(config)
        num_batch_size = config.training.batch_size
        config_.training.batch_size = config.prune_cfg.est_ranges_batch_size
        training_args_ = _make_huggingface_training_args(config_)

        _, datasets_shared, train_dataset_range, _, _ = _make_datasets_and_trainer(
            config, model, model_enum, tokenizer, task_data, processor, training_args_,
        )

        # estimate (FP32) ranges for per-group act prune permutation:
        num_loader = len(train_dataset_range) if train_dataset_range is not None else 1
        # print('num_loader:',num_loader)
        # import pdb;pdb.set_trace()
        num_steps_per_epoch = math.ceil((num_loader) / (num_batch_size * training_args_.gradient_accumulation_steps))
        if training_args_.num_train_epochs != 0:
            # num_steps_per_epoch = math.ceil((num_loader) / (num_batch_size * training_args_.gradient_accumulation_steps))
            num_steps_per_epoch = math.ceil((num_loader) / (num_batch_size))
            all_steps = training_args_.num_train_epochs * num_steps_per_epoch
        else:
            assert training_args_.max_steps is not None, 'max_steps must be set if num_epochs is 0'
            # all_steps = training_args_.max_steps / training_args_.gradient_accumulation_steps
            all_steps = training_args_.max_steps * training_args_.gradient_accumulation_steps
                
        logger.info(f'**************** all steps: {all_steps} ****************')
        
        model = set_minmax_prune_ratio(model, max_prune_ratio=config.prune_cfg.max_prune_ratio, min_prune_ratio=config.prune_cfg.min_prune_ratio)
        # model = _prepare_pruned_model(
        #     config, model, loader=loader
        # )
        print(config.prune_cfg.max_prune_ratio, config.prune_cfg.min_prune_ratio)
        
        if config.prune_cfg.mag_prune == True:
            model.set_mag_prune()
            # model.set_hand_ratio(config.prune_cfg.hand_ratio)
            model = set_hand_ratio(model, hand_ratio=config.prune_cfg.hand_ratio)
        if config.prune_cfg.fix_prob == True:
            model.set_fix_prob()
        if config.prune_cfg.hard == True:
            logger.info(f'**************** set hard gumbel all ****************')
            model.set_gumbel_hard()
        if config.prune_cfg.only_size_hard == True:
            logger.info(f'**************** set hard gumbel search in size ****************')
            model.set_only_size_hard()
            
        if config.prune_cfg.decay_tau == True:
            logger.info(f'**************** set decay tau from 0.5 to 0.01 ****************')
            model.set_decay_tau()

        model = set_tau_min_max(model, eta_max=config.prune_cfg.eta_max, eta_min=config.prune_cfg.eta_min)
        # model.set_total_steps(all_steps)
        model = set_total_steps(model, all_steps) 
        # print('current_model_total_steps:', model.total_steps)
        """model = set_prune_ratio(model, config.prune_cfg.max_prune_ratio)
        model = set_total_steps(model, all_steps)
        model = set_tau_min_max(model) # NOTE:wait to add
        # model = set_prune_trans(model)
        print('modified_model_total_steps:', model.total_steps, model.lambda1, model.lambda2, model.prune_ratio)"""
        # # model.precision_levels = [8,4] # [0,1]
        # logger.info('after MP model:')
        # logger.info(model)

    elif _mag_struct_finetune:
        logger.info(
            "mag_prune_first structural: skip pruner prep; prefetch datasets for Trainer"
        )
        gacc = int(config.training.gradient_accumulation_steps or 1)
        max_steps = int(config.training.max_steps or 0)
        all_steps = max(max_steps * gacc, 1)
        model = set_tau_min_max(
            model, eta_max=config.prune_cfg.eta_max, eta_min=config.prune_cfg.eta_min
        )
        model = set_total_steps(model, all_steps)
        model = set_warmup_ratio(model, warmup_ratio=0.1)
        _, datasets_shared, _, _, _ = _make_datasets_and_trainer(
            config, model, model_enum, tokenizer, task_data, processor, training_args,
        )
    else:
        logger.info('original full precision model:')
        logger.info(model)

    # make datasets and Trainer
    _datasets_kw = {}
    if datasets_shared is not None:
        _datasets_kw["datasets_prepared"] = datasets_shared
    trainer, datasets, train_dataset, eval_dataset, processor = _make_datasets_and_trainer(
        config, model, model_enum, tokenizer, task_data, processor, training_args, **_datasets_kw
    )
    # import pdb;pdb.set_trace()
    # task_data_test = load_task_data(data_dir=config.benchmark.data_dir, test_data='all')
    # test_datasets_all = _make_datasets_for_test(task_data_test, processor, return_att_mask=False)

    
    ## attach some helper attributes for TB, saving, logging etc.
    # TB counters
    
    # Training!
    model_name_or_path = model_data.model_name_or_path
    
    # act_prune = config.prune_cfg.act_prune and (config.prune_cfg.fixed_8bit or config.prune_cfg.fixed_48bit or not config.prune_cfg.prune_only)
    
    # model.set_prune_state(config.prune_cfg.weight_prune, act_prune)
    
    # per_layer_macs, pruner_groups = get_groups_and_macs(model, 'speech', 'wav2vec2')
    # relevant_pruners = get_relevant_pruners_from_groups(pruner_groups)
    
    # all_pruned = config.prune_cfg.weight_prune and config.prune_cfg.act_prune
    # flops, mac, params = get_macs(model.encoder.0.k_proj)
    # set_macs(model)
    # mac_dict = return_dict()
    # import pdb;pdb.set_trace()
    # if config.prune_cfg.reg_type == "bop":
    #     assign_macs_wav(
    #         mac_dict=mac_dict, prune_only=config.prune_cfg.prune_only, model=model
    #     )
        
    # if all_pruned:
    #     pretty_print_pruning_wav(
    #        model, mac_dict, prune_only=config.prune_cfg.prune_only
    #     )
    #     print_bitops_wav(
    #         model,
    #         logfile=None,
    #         prune_only=config.prune_cfg.prune_only,
    #     )
    
    # assert model.calib == config.act_prune.num_batches if config.prune_cfg.act_prune else 1
    
    if config.training.do_train:
        logger.info('*** Training ***')
        _resume_ckpt = None
        if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
            if os.path.isfile(os.path.join(model_name_or_path, "trainer_state.json")):
                _resume_ckpt = model_name_or_path
        trainer.train(resume_from_checkpoint=_resume_ckpt)
        
        _mag_prune_first = bool(
            getattr(getattr(config, "prune_cfg", None), "mag_prune_first", False)
        )
        if config.progress.save_model and (
            config.training.num_epochs != 0 or _mag_prune_first
        ):
            trainer.save_model()  # saves the tokenizer too

        if (
            "prune_cfg" in config
            and getattr(config.prune_cfg, "channel_pruning", False)
            and not _mag_prune_first
        ):
            try:
                from structural_prune_export import (
                    export_structural_prune_posttrain,
                    load_structural_pruned_for_inference,
                )

                pruned_dir = export_structural_prune_posttrain(
                    trainer.model, trainer.model.config, trainer, config
                )
                if pruned_dir:
                    _dev = next(trainer.model.parameters()).device
                    model = load_structural_pruned_for_inference(pruned_dir, device=_dev)
                    trainer.model = model
                    logger.info(
                        "structural_prune_export: using shape-reduced ckpt from %s for eval",
                        pruned_dir,
                    )
            except Exception:
                logger.exception("structural_prune_export failed after train-pruned")

    # fix ranges after training, for final evaluation
    # if 'prune' in config:
    model.eval()
    # model.fix_ranges()
    trainer.model.eval()
    # trainer.model.fix_ranges()
    # import pdb;pdb.set_trace()
    # Validation!
    final_score = None
    if config.training.do_eval:
        logger.info('*** Evaluation ***')

        # if AdaRound, evaluate with multiple range settings for activations
    
        # no adaround
        logger.info("no adaround:")
        # eval_wer = True
        eval_wer = not _smoke_eval_test_only()
        if eval_wer:
            final_score_eval = _eval_task(config, trainer, eval_dataset, datasets, model, processor)
            logger.info(f'Eval clean ASR -> {100. * final_score_eval:.10f}')
            # if config.training.do_train:
            with open(os.path.join(config.base.output_dir, 'final_score.txt'), 'a') as f:
                f.write(f'Eval clean ASR -> {100. * final_score_eval:.10f}\n')
        test_wer = True
        if test_wer:
            # import pdb;pdb.set_trace()
            WER_result_val_other, WER_result_test_other, WER_result_test_clean = _test_task(config, datasets, model, processor)
            logger.info(f'Test Clean ASR -> {100. * WER_result_test_clean:.10f}')
            logger.info(f'Test Other ASR -> {100. * WER_result_test_other:.10f}')
            if not _smoke_eval_test_only():
                logger.info(f'Eval Other ASR -> {100. * WER_result_val_other:.10f}')
            # if config.training.do_train:
            with open(os.path.join(config.base.output_dir, 'final_score.txt'), 'a') as f:
                f.write(f'Test Clean ASR -> {100. * WER_result_test_clean:.10f}\n')
                f.write(f'Test Other ASR -> {100. * WER_result_test_other:.10f}\n')
                if not _smoke_eval_test_only():
                    f.write(f'Eval Other ASR -> {100. * WER_result_val_other:.10f}\n')

    return final_score


def _eval_task(config, trainer, eval_dataset, datasets ,model, processor):
    # loop to handle MNLI double evaluation (matched and mis-matched accuracy)

    # model.precision_levels = [2]
    # subtask_names = [task.name]
    eval_datasets = [eval_dataset]

    for eval_dataset in eval_datasets:
            

        model.eval()
        trainer.model.eval()
        logger.info('trainer_eval:')
        # eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        # logger.info(eval_result)

        wer_metric = _load_wer_metric()
        logger.info('WER_Metric:')

        _log_encoder_prune_ratio_before_wer_map(
            config,
            model,
            eval_dataset,
            phase="inference_pre_map_eval_clean",
        )

        """validation clean"""
        _map_to_result = partial(map_to_result, model=model, processor=processor, return_att_mask=False)
        # import pdb;pdb.set_trace()
        results = eval_dataset.map(_map_to_result, remove_columns=eval_dataset.column_names, load_from_cache_file=False)

        print("***** Eval clean WER *****")
        WER_result = wer_metric.compute(predictions=results["pred_str"], references=results["text"])
        
        result_dir = os.environ.get(
            "SSLPRUNE_DECODE_DIR",
            os.path.join(config.base.output_dir, "decode"),
        )
        
        output_dir = config.base.output_dir.split('/')[-1]
        if re.match(r'checkpoint-\d+', output_dir):
            output_dir = config.base.output_dir.split('/')[-3] + '_' + config.base.output_dir.split('/')[-1]
        
        output_dir = config.base.output_dir.split('/')[-4] + '_' + config.base.output_dir.split('/')[-3] + '_' + config.base.output_dir.split('/')[-2] + '_' + config.base.output_dir.split('/')[-1]
            
        """if not os.path.exists(os.path.join(result_dir, 'eval_clean_ref.txt')):
            with open(os.path.join(result_dir, 'eval_clean_ref.txt'), 'w') as f:
                for i in range(len(results["text"])):
                    f.write(results["text"][i] + ' ' + f'({i}.txt)'  + '\n')"""

        save_path = result_dir + '/' + output_dir + '_' + 'last' + '_'
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        
        with open(os.path.join(save_path, f'eval_clean_decode.txt'), 'w') as f:
            for i in range(len(results["text"])):
                f.write(results["pred_str"][i] + ' ' + f'({i}.txt)'  + '\n')
                
        logger.info('Eval clean WER:{:.10f}'.format(WER_result))
            
    return WER_result

def _test_task(config, datasets_all ,model, processor):
    # loop to handle MNLI double evaluation (matched and mis-matched accuracy)
    # model.precision_levels = [0]
    # subtask_names = [task.name]
    # import pdb;pdb.set_trace()
    wer_metric = _load_wer_metric()
    logger.info('WER_Metric:')
    
    WER_result_val_other, WER_result_test_other, WER_result_test_clean = 0, 0, 0
    _map_to_result = partial(map_to_result, model=model, processor=processor, return_att_mask=False)

    _pre_map_ds = None
    for _k in ("test_clean", "test_other", "val_other"):
        if _k in datasets_all.keys() and len(datasets_all[_k]) > 0:
            _pre_map_ds = datasets_all[_k]
            break
    if _pre_map_ds is not None:
        _log_encoder_prune_ratio_before_wer_map(
            config,
            model,
            _pre_map_ds,
            phase="inference_pre_map_test",
        )

    if 'test_clean' in datasets_all.keys():
        
        results_test_clean = datasets_all['test_clean'].map(_map_to_result, remove_columns=datasets_all['test_clean'].column_names, load_from_cache_file=False)
        print("***** Test clean WER *****")

        result_dir = _decode_export_dir(config)
        save_path = os.path.join(result_dir, _decode_run_subdir(config))
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'test_clean_decode.txt'), 'w') as f:
            for i in range(len(results_test_clean["text"])):
                f.write(results_test_clean["pred_str"][i] + ' ' + f'({i}.txt)'  + '\n')
                
        WER_result_test_clean = wer_metric.compute(predictions=results_test_clean["pred_str"], references=results_test_clean["text"])
        
        logger.info('Test clean WER:{:.10f}'.format(WER_result_test_clean))
        
    if not _smoke_eval_test_only() and 'val_other' in datasets_all.keys():
        
        results_val_other = datasets_all['val_other'].map(_map_to_result, remove_columns=datasets_all['val_other'].column_names, load_from_cache_file=False)

        print("***** Eval other WER *****")
        WER_result_val_other = wer_metric.compute(predictions=results_val_other["pred_str"], references=results_val_other["text"])
        
        result_dir = _decode_export_dir(config)
        save_path = os.path.join(result_dir, _decode_run_subdir(config))
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'eval_other_decode.txt'), 'w') as f:
            for i in range(len(results_val_other["text"])):
                f.write(results_val_other["pred_str"][i] + ' ' + f'({i}.txt)'  + '\n')
                
        logger.info('Eval other WER:{:.10f}'.format(WER_result_val_other))
        
    if 'test_other' in datasets_all.keys():
        
        results_test_other = datasets_all['test_other'].map(_map_to_result, remove_columns=datasets_all['test_other'].column_names, load_from_cache_file=False)

        print("***** Test other WER *****")
        WER_result_test_other = wer_metric.compute(predictions=results_test_other["pred_str"], references=results_test_other["text"])
        
        result_dir = _decode_export_dir(config)
        save_path = os.path.join(result_dir, _decode_run_subdir(config))
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'test_other_decode.txt'), 'w') as f:
            for i in range(len(results_test_other["text"])):
                f.write(results_test_other["pred_str"][i] + ' ' + f'({i}.txt)'  + '\n')        
        logger.info('Test other WER:{:.10f}'.format(WER_result_test_other))
            
    return WER_result_val_other, WER_result_test_other, WER_result_test_clean

def _run(config):
    # import pdb;pdb.set_trace()
    """Common routine to run training/validation on a set of tasks."""
    do_train = config.training.do_train
    mode_str = 'Training' if do_train else 'Validating'

    # parse tasks
    # task_flag = BENCHMARK_Task.from_str(*config.benchmark.task)
    # logger.info(f'{mode_str} on tasks: {list(task_flag.iter_names())}')

    
    # for task in task_flag.iter():
        # logger.info(f'{mode_str} on task {task.name} ...')
        # prepare task-specific config, if necessary
    if config.model.model_path is None:  # use pre-trained backbone for training/validation
        task_config = config
    else:
        # load the suitable checkpoint
        if do_train:
            # simply load the checkpoint given by --model-path
            task_config = config
        else:
            # for validation, load the checkpoint from the corresponding subfolder given by task
            task_dirpath = Path(config.model.model_path)
            task_out_dirpaths = task_dirpath.glob('**')
            non_empty_task_out_dirpaths = list(filter(_is_non_empty_dir, task_out_dirpaths))
            if not len(non_empty_task_out_dirpaths):
                raise RuntimeError(f'Task directory ({task_dirpath}) is empty.')
            if len(non_empty_task_out_dirpaths) > 1:
                msg = [f'Task directory ({task_dirpath}) contains multiple checkpoints:']
                for dirpath in non_empty_task_out_dirpaths:
                    msg.append(f'* {dirpath}')
                raise RuntimeError('\n'.join(msg))

            task_out_dirpath = str(non_empty_task_out_dirpaths[0])
            task_config = deepcopy(config)
            if config.base.output_dir is None:
                task_config.base.output_dir = task_out_dirpath
            task_config.model.model_path = task_out_dirpath

    # load data
    logger.info(f'data_dir: {task_config.benchmark.data_dir}  ...\n')

    task_data = load_task_data(data_dir=task_config.benchmark.data_dir, data_type=task_config.data.type, test_data=None)

    model_data = load_model_and_tokenizer(**task_config.model)

    _run_task(task_config, task_data, model_data)


    # log elapsed timexr
    # logger.info(s.format())


def _train(config):
    # check and set training-specific options
    if config.base.output_dir is None:
        raise ValueError('--output-dir must be provided for training')
    config.training.do_train = True
    _run(config)


def _validate(config):

    config.base.overwrite_output = False
    config.training.do_eval = True
    config.training.do_train = False

    _run(config)


@benchmark.command()
@pass_config
@benchmark_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
def train_baseline(config):
    _train(config)


@benchmark.command()
@pass_config
@benchmark_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
@pruning_options
@transformer_prune_options
def train_pruned(config):
    _train(config)


@benchmark.command()
@pass_config
@benchmark_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
def validate_baseline(config):
    _validate(config)


@benchmark.command()
@pass_config
@benchmark_options
@transformer_base_options
@transformer_data_options
@transformer_model_options
@transformer_training_options
@transformer_progress_options
@pruning_options
@transformer_prune_options
def validate_pruned(config):
    _validate(config)


if __name__ == '__main__':
    benchmark()
