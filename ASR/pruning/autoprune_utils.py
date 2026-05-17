# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import warnings
import torch
from torch.nn import functional as F
from torch import nn
from torch.nn.modules.pooling import _AdaptiveAvgPoolNd, _AvgPoolNd

from pruning.base_pruned_classes import FP32Acts, PrunedActivation, PrunedModule
from pruning.hijacker import PruningHijacker, activations_list

from pruning_manager_direct import PruningManager
from pruning.pruners_direct import AsymmetricUniformPruner, SymmetricUniformPruner, PMethods, PrunerNotInitializedError

class PruneConv1d(PruningHijacker, nn.Conv1d): # activation=None?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def run_forward(self, x, weight, bias, offsets=None):
        
        return F.conv1d(x.contiguous(), weight.contiguous(), bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

use_L2_2048, use_L2_2560, use_L2_2304, use_select = 1,1,1,0


class PruneLinear(PruningHijacker, nn.Linear):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_forward(self, x, weight, bias, offsets=None):
       
        return F.linear(x.contiguous(), weight.contiguous(), bias=bias)
        # if (weight.size()[-1] == 768 and weight.size()[-2] == 768) or (weight.size()[-1] == 1024 and weight.size()[-2] == 1024):
        if (weight.size()[-1] == 32) or (weight.size()[-2] == 32):
            return F.linear(x.contiguous(), weight.contiguous(), bias=bias), weight
        else:
            return F.linear(x.contiguous(), weight.contiguous(), bias=bias)




class PrunedActivationWrapper(PrunedActivation):
    """
    Wraps over a layer and pruned the activation.
    It also allow for tying the input and output pruner which is helpful
    for layers such Average Pooling.
    """
    def __init__(self, layer, tie_activation_pruners=False,
                 input_pruner: PruningManager = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tie_activation_pruners = tie_activation_pruners
        if input_pruner:
            assert isinstance(input_pruner, PruningManager)
            self.activation_pruner = input_pruner
        self.layer = layer

    def prune_activations_no_range_update(self, x):
        if self._prune_a:
            return self.activation_pruner.pruner(x)
        else:
            return x

    def forward(self, x):
        x = self.layer(x)
        if self.tie_activation_pruners:
            # The input activation pruner is used to prune the activation
            # but without updating the pruning range
            return self.prune_activations_no_range_update(x)
        else:
            return self.prune_activations(x)


class PruneGroupNorm(PruningHijacker, nn.GroupNorm): # there are params to be learned
    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
    def run_forward(self, x, weight, bias, offsets=None):

        return F.group_norm(
        input=x.contiguous(),
        num_groups=self.num_groups,
        weight=weight.contiguous(),
        bias=bias.contiguous(),
        eps=self.eps,
        )
    
class PruneLayerNorm(PruningHijacker, nn.LayerNorm): # there are params to be learned
    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
      
    def run_forward(self, x, weight, bias, offsets=None):
       
        return F.layer_norm(
            input=x.contiguous(),
            normalized_shape=self.normalized_shape,
            weight=weight.contiguous(),
            bias=bias.contiguous(),
            eps=self.eps,
        )

class PruneEmbedding(PruningHijacker, nn.Embedding):
    def __init__(self, *args, activation=None, **kwargs):
        super().__init__(*args, activation=activation, **kwargs)
        # NB: Embedding should not prune activations, as it is simply a lookup table,
        # which is already pruned.
        self.activation_pruner = FP32Acts()

    def run_forward(self, x, weight, bias, offsets=None):
        return F.embedding(
            input=x.contiguous(),
            weight=weight.contiguous(),
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


module_map = {
    nn.Conv1d: PruneConv1d,
    nn.Linear: PruneLinear,
    nn.GroupNorm: PruneGroupNorm,
    nn.LayerNorm: PruneLayerNorm,
    nn.Embedding: PruneEmbedding
}


non_param_modules = (_AdaptiveAvgPoolNd, _AvgPoolNd)


def get_act(module, i):
    result, act_idx = None, None
    for i in range(i + 1, len(module)):
        if isinstance(module[i], tuple(activations_list)):
            result = module[i]
            act_idx = i
            break
    return result, act_idx

def get_conv1d_args(module):
    args = dict(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        padding=module.padding,
        groups=module.groups,
        stride=module.stride,
        bias=module.bias is not None,
    )
    return args

def get_linear_args(module):
    args = dict(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
    )
    return args


def get_layernorm_args(module):
    args = dict(normalized_shape=module.normalized_shape, eps=module.eps)
    return args

def get_groupnorm_args(module):
    args = dict(num_channels=module.num_channels, num_groups=module.num_groups, eps=module.eps)
    return args

def get_embedding_args(module):
    args = dict(
        num_embeddings=module.num_embeddings,
        embedding_dim=module.embedding_dim,
        padding_idx=module.padding_idx,
        max_norm=module.max_norm,
        norm_type=module.norm_type,
        scale_grad_by_freq=module.scale_grad_by_freq,
        sparse=module.sparse,
    )
    return args


def get_module_args(mod, act):
    if isinstance(mod, nn.Linear):
        kwargs = get_linear_args(mod)
    elif isinstance(mod, nn.LayerNorm):
        kwargs = get_layernorm_args(mod)
    elif isinstance(mod, nn.Embedding):
        kwargs = get_embedding_args(mod)
    elif isinstance(mod, nn.Conv1d):
        kwargs = get_conv1d_args(mod)
    elif isinstance(mod, nn.GroupNorm):
        kwargs = get_groupnorm_args(mod)
    else:
        raise ValueError

    kwargs['activation'] = act
    return kwargs


def prune_module(module, i, **prune_params):
    act, act_idx = get_act(module, i)
    modtype = module_map[type(module[i])]

    kwargs = get_module_args(module[i], act)
    new_module = modtype(**kwargs, **prune_params)
    new_module.weight.data = module[i].weight.data.clone()

    if module[i].bias is not None:
        new_module.bias.data = module[i].bias.data.clone()

    return new_module, i + int(bool(act)) + 1


def prune_sequence(model, specials=None, tie_activation_pruners=False, **prune_params):
    specials = specials or dict()

    i = 0
    prune_modules = []
    while i < len(model):
        if isinstance(model[i], PrunedModule):
            prune_modules.append(model[i])
        elif type(model[i]) in module_map:
            new_module, new_i = prune_module(model, i, **prune_params)
            prune_modules.append(new_module)
            i = new_i
            continue

        elif type(model[i]) in specials:
            prune_modules.append(specials[type(model[i])](model[i], **prune_params))

        elif isinstance(model[i], non_param_modules):
            # check for last pruner
            input_pruner = None
            if (
                prune_modules
                and isinstance(prune_modules[-1], PrunedModule)
                and tie_activation_pruners
            ):
                input_pruner = prune_modules[-1].activation_pruner
                warnings.warn(
                    f'Tying input pruner {i}^th layer of type '
                    f'{type(prune_modules[-1])} to the pruned {type(model[i])} '
                    f'following it'
                )
            prune_modules.append(
                PrunedActivationWrapper(
                    model[i],
                    tie_activation_pruners=tie_activation_pruners,
                    input_pruner=input_pruner,
                    **prune_params,
                )
            )

        else:
            prune_modules.append(prune_model(model[i], specials=specials, **prune_params))
        i += 1
    return prune_modules


def prune_sequential(model, specials=None, tie_activation_pruners=False, **prune_params):
    prune_modules = prune_sequence(model, specials, tie_activation_pruners, **prune_params)
    return nn.Sequential(*prune_modules)


def prune_module_list(model, specials=None, tie_activation_pruners=False, **prune_params):
    prune_modules = prune_sequence(model, specials, tie_activation_pruners, **prune_params)
    return nn.ModuleList(prune_modules)


def prune_model(model, prune_output=torch.tensor(1.), specials=None, tie_activation_pruners=False, layer_name = None,  **prune_params):
    specials = specials or dict()
    include_pruning = prune_params['include_pruning']
    channel_pruning = prune_params['channel_pruning']
    
    if isinstance(model, nn.Sequential):
        prune_model = prune_sequential(
            model, specials, tie_activation_pruners, **prune_params
        )

    elif type(model) in specials:
        prune_model = specials[type(model)](model, **prune_params)

    elif isinstance(model, non_param_modules):
        prune_model = PrunedActivationWrapper(model, **prune_params)

    elif type(model) in module_map:
        # if we do isinstance() then we might run into issues with modules that inherit from
        # one of these classes, for whatever reason
        # import pdb;pdb.set_trace()
        modtype = module_map[type(model)]
        kwargs = get_module_args(model, None)

        prune_model = modtype(**kwargs, **prune_params)

        prune_model.weight.data = model.weight.data

        # prune_model.weight_pruneizer_saspg.pruner.pruner.mask_scores = torch.nn.Parameter(torch.empty(prune_model.weight.size()))
        
        if channel_pruning:
            prune_model.weight_pruneizer_saspg.pruner.pruner.prune_output = prune_output
        
        if include_pruning:
            prune_model.weight_pruneizer.pruner.pruner.x_prune_mask = torch.nn.Parameter(torch.abs(model.weight.data))
            
            # prune_model.weight_pruneizer.pruner.pruner.x_prune_mask = torch.nn.Parameter(model.weight.data)
            print('init self.x_prune_mask ...')
        # prune_model.weight_pruneizer.pruner.pruner.x_prune_mask = torch.nn.Parameter(torch.abs(model.weight.data))
        if layer_name:
            if 'q_proj' in layer_name or 'k_proj' in layer_name or 'v_proj' in layer_name or 'intermediate_dense' in layer_name:
                prune_model.weight_pruneizer_saspg.pruner.pruner.layer_name = layer_name
                prune_model.weight_pruneizer_saspg.pruner.pruner.channel_vector = torch.nn.Parameter(torch.ones(prune_model.weight.shape[1]), requires_grad=True)
    
                # prune_model.threshold_prune_channel = torch.nn.Parameter() # this will conflict with the defination in hijacker
            elif 'out_proj' in layer_name or 'output_dense' in layer_name:
                prune_model.weight_pruneizer_saspg.pruner.pruner.layer_name = layer_name
                prune_model.weight_pruneizer_saspg.pruner.pruner.channel_vector = torch.nn.Parameter(torch.ones(prune_model.weight.shape[0]), requires_grad=True)
        
            
        
        if getattr(model, 'bias', None) is not None:
            prune_model.bias.data = model.bias.data

    else:
        # unknown type, try to prune all child modules
        prune_model = copy.deepcopy(model)
        for name, module in prune_model._modules.items():
            new_model = prune_model(module, specials=specials, **prune_params)
            if new_model is not None:
                setattr(prune_model, name, new_model)

    return prune_model
