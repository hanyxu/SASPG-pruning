# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from enum import Flag, auto
from functools import reduce
from operator import or_

from datasets import DatasetDict, load_dataset

from utils.librispeech_data import csv_metadata_dir, csv_path, resolve_audio_path, train_csv_name
from utils.utils import DotDict

logger = logging.getLogger("BENCHMARK")
logger.setLevel(logging.INFO)


class BENCHMARK_Task(Flag):
    wav2vec2 = auto()
    hubert = auto()
    test = auto()
    wavlm = auto()
    all = wav2vec2 | hubert | wavlm | test

    def __contains__(self, item):
        return (self.value & item.value) == item.value

    @classmethod
    def from_str(cls, *names):
        assert len(names)
        return reduce(or_, map(cls.__getattr__, names))

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]

    def iter(self):
        for x in self.__class__.__members__.values():
            if x in self and x != self.__class__.all:
                yield x

    def iter_names(self):
        for x in self.iter():
            yield x.name


TASK_TO_FINAL_METRIC = {
    BENCHMARK_Task.wav2vec2: "wer",
    BENCHMARK_Task.hubert: "wer",
    BENCHMARK_Task.wavlm: "wer",
}


def _resolve_file_column(batch):
    batch["file"] = resolve_audio_path(batch["file"])
    return batch


def load_task_data(data_dir, data_type, test_data=None):
    del data_dir
    out = DotDict()
    if test_data:
        raise NotImplementedError("test_data splits are not used in this release.")

    data_files = {
        "train": csv_path(train_csv_name(data_type)),
        "validation": csv_path("dev-clean.csv"),
        "val_other": csv_path("dev-other.csv"),
        "test_other": csv_path("test-other.csv"),
        "test_clean": csv_path("test-clean.csv"),
    }
    logger.info("Loading LibriSpeech manifests from %s", csv_metadata_dir())
    datasets = load_dataset("csv", data_files=data_files)
    _np = int(os.environ.get("LIBRISPEECH_PATH_RESOLVE_NUM_PROC", "8"))
    if _np > 0:
        datasets = datasets.map(_resolve_file_column, num_proc=min(_np, 32))
    else:
        datasets = datasets.map(_resolve_file_column)
    out.datasets = datasets
    return out
