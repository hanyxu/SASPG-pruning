# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic
    # torch.use_deterministic(True)
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    # torch.use_deterministic_algorithms(True)


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))


def count_embedding_params(model):
    return sum(count_params(m) for m in model.modules() if isinstance(m, nn.Embedding))

def count_conv1d_params(model):
    return sum(count_params(m) for m in model.modules() if isinstance(m, nn.Conv1d))

def count_layernorm_params(model):
    return sum(count_params(m) for m in model.modules() if isinstance(m, nn.LayerNorm))

def count_Groupnorm_params(model):
    return sum(count_params(m) for m in model.modules() if isinstance(m, nn.GroupNorm))

def count_linear_params(model):
    return sum(count_params(m) for m in model.modules() if isinstance(m, nn.Linear))

def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


def kl_divergence(P, Q):
    if isinstance(P, list):
        assert isinstance(Q, list) and len(P) == len(Q)
        kl = 0
        bs = 0
        for i in range(len(P)):
            kl += (P[i] * (P[i] / Q[i]).log()).sum()
            if len(P[i].size()) == 2:
                bs += P[i].size(0)
            elif len(P[i].size()) == 3:
                bs += P[i].size(0) * P[i].size(1)
        return kl / bs
    else:
        if len(P.size()) == 2:
            return (P * (P / Q).log()).sum() / P.size(0) # batch size
        elif len(P.size()) == 3:
            return (P * (P / Q).log()).sum() / P.size(0) / P.size(1)
    # F.kl_div(Q.log(), P, None, None, 'sum')

def symmetric_kl(P, Q):
    return (kl_divergence(P, Q) + kl_divergence(Q, P)) / 2



def pass_data_for_range_estimation(
    loader, model, act_prune, weight_prune, max_num_batches=20, cross_entropy_layer=None, inp_idx=0
):
    # import pdb;pdb.set_trace()
    model.set_prune_state(weight_prune, act_prune)
    model.eval()
    # import pdb;pdb.set_trace()
    if cross_entropy_layer is not None:
        layer_xent = get_layer_by_name(model, cross_entropy_layer)
        if layer_xent:
            print(f'Set cross entropy estimator for layer "{cross_entropy_layer}"')
            act_prune_mgr = layer_xent.activation_pruner
            act_prune_mgr.range_estimator = RangeEstimators.cross_entropy.cls(
                per_channel=act_prune_mgr.per_channel,
                pruner=act_prune_mgr.pruner,
                **act_prune_mgr.init_params,
            )
        else:
            raise ValueError('Cross-entropy layer not found')

    model.to('cuda')
    device = next(model.parameters()).device
    print('range estimation model on device:',device)
    # import pdb;pdb.set_trace()
    # print('current:',model.precision_levels)
    # # if hasattr(model,'precision_levels'):
    # model.precision_levels = [8, 4] # [0,1]

    # model.precision_levels = [4] # [0,1]
    model.precision_levels = [8, 4, 2]
    # model.precision_levels = [8,6]

    # model.precision_levels = [8, 4]

    # model.precision_levels = [8,6,4]

    # model.precision_levels = [6]
    # model.precision_levels = [30,25,23]
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.device}")

    for i, data in enumerate(loader):
        # import pdb;pdb.set_trace()
        print('batch:',i)
        try:
            if isinstance(data, (tuple, list)):
                x = data[inp_idx].to(device=device)
                model(x)
            else:
                x = {k: v.to(device=device) for k, v in data.items()}
                # import pdb;pdb.set_trace()
                # 默认是0，学习min max
                model(**x) # 这里就会调用range
                
        except StopForwardException:
            pass
    # 这里还没有init
        if i >= max_num_batches - 1 or not act_prune: # batches就是est_num_batches
            break
    # model.precision_levels = [8,6]


def pass_data_for_range_estimation_width(
    loader, model, act_prune, weight_prune, max_num_batches=20, cross_entropy_layer=None, inp_idx=0
):
    # import pdb;pdb.set_trace()
    model.set_prune_state(weight_prune, act_prune)
    model.eval()
    # import pdb;pdb.set_trace()
    if cross_entropy_layer is not None:
        layer_xent = get_layer_by_name(model, cross_entropy_layer)
        if layer_xent:
            print(f'Set cross entropy estimator for layer "{cross_entropy_layer}"')
            act_prune_mgr = layer_xent.activation_pruner
            act_prune_mgr.range_estimator = RangeEstimators.cross_entropy.cls(
                per_channel=act_prune_mgr.per_channel,
                pruner=act_prune_mgr.pruner,
                **act_prune_mgr.init_params,
            )
        else:
            raise ValueError('Cross-entropy layer not found')

    device = next(model.parameters()).device
    # import pdb;pdb.set_trace()
    # print('current:',model.precision_levels)
    # # if hasattr(model,'precision_levels'):
    # model.precision_levels = [8,4] # [0,1]
    # model.precision_levels = [8,6,4]
    # model.precision_levels = [30,23,20]
    model.precision_levels = [30, 25]
    # model.precision_levels = [30, 25, 23, 20]
    # model.precision_levels = [30, 20]
    # model.precision_levels = [30, 23]
    # model.precision_levels = [23]
    # model.precision_levels = [20]



    
    for i, data in enumerate(loader):
        # import pdb;pdb.set_trace()
        print('batch:',i)
        try:
            if isinstance(data, (tuple, list)):
                x = data[inp_idx].to(device=device)
                model(x)
            else:
                x = {k: v.to(device=device) for k, v in data.items()}
                # import pdb;pdb.set_trace()
                # 默认是0，学习min max
                model(**x) # 这里就会调用range
                
        except StopForwardException:
            pass
    # 这里还没有init
        if i >= max_num_batches - 1 or not act_prune: # batches就是est_num_batches
            break
    # model.precision_levels = [8,6]
        
class DotDict(dict):
    """
    A dictionary that allows attribute-style access.

    Examples
    --------
    >>> config = DotDict(a=None)
    >>> config.a = 42
    >>> config.b = 'egg'
    >>> config  # can be used as dict
    {'a': 42, 'b': 'egg'}
    """
    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")


class Stopwatch:
    """
    A simple cross-platform context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as st:
    ...     time.sleep(0.101)  #doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    """
    def __init__(self, name=None, verbose=False):
        self._name = name
        self._verbose = verbose

        self._start_time_point = 0.0
        self._total_duration = 0.0
        self._is_running = False

        if sys.platform == 'win32':
            # on Windows, the best timer is time.clock()
            self._timer_fn = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_fn = time.time

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._verbose:
            self.print()

    def start(self):
        if not self._is_running:
            self._start_time_point = self._timer_fn()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total_duration += self._timer_fn() - self._start_time_point
            self._is_running = False
        return self

    def reset(self):
        self._start_time_point = 0.0
        self._total_duration = 0.0
        self._is_running = False
        return self

    def _update_state(self):
        now = self._timer_fn()
        self._total_duration += now - self._start_time_point
        self._start_time_point = now

    def _format(self):
        prefix = f'[{self._name}]' if self._name is not None else 'Elapsed time'
        info = f'{prefix}: {self._total_duration:.3f} sec'
        return info

    def format(self):
        if self._is_running:
            self._update_state()
        return self._format()

    def print(self):
        print(self.format())

    def get_total_duration(self):
        if self._is_running:
            self._update_state()
        return self._total_duration


