# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

# from pruning.base_pruned_classes import FP32Acts


def _hijack_act_prune(module, value):
    # import pdb;pdb.set_trace()
    if value is None:
        return

    if isinstance(value, int):
        module.activation_pruner.pruner.n_bits = value
        print(f'weight_pruneizer to {value}-bit')
    elif value == 'fp32':
        module.activation_pruner = FP32Acts()
        print(f'weight_pruneizer to FP32')
    elif value == 'per_embd':
        set_act_prune_axis_and_groups(module, axis=2, n_groups=None)
        print(f'weight_pruneizer to per_embd')
    elif value.startswith('ngp'):
        set_act_prune_axis_and_groups(module, axis=2, n_groups=int(value[3:]), permute=True)
        print(f'weight_pruneizer to ngp')
    elif value.startswith('ng'):
        set_act_prune_axis_and_groups(module, axis=2, n_groups=int(value[2:]), permute=False)
        print(f'weight_pruneizer to ng')
    else:
        raise NotImplementedError(f'Unknown value "{value}" in prune_dict')


def _hijack_weight_prune(module, value):
    if value is None:
        return
    # import pdb;pdb.set_trace()
    if isinstance(value, int):
        module.weight_pruneizer.pruner.n_bits = value
        print(f'weight_pruneizer to {value}-bit')
    elif value == 'fp32':
        module.weight_pruneizer = FP32Acts()
        print(f'weight_pruneizer to FP32')
    else:
        raise NotImplementedError(f'Unknown value "{value}" in prune_dict')


def hijack_act_prune(prune_dict, name, m):
    # import pdb; pdb.set_trace()
    value = prune_dict.get(name, None)
    _hijack_act_prune(m, value)
    


def hijack_weight_prune(prune_dict, name, m):
    value = prune_dict.get(name, None)
    _hijack_weight_prune(m, value)

def hijack_weight_prune_modules(prune_dict, name, m):
    # import pdb; pdb.set_trace()
    value = prune_dict.get(name, None)
    if value is not None:
        print(f'Mixed weight command {name}')
    for title, m_ in m.named_modules():
        if hasattr(m_, 'weight_pruneizer'):
            if value is not None:
                print(f'Mixed weight pruned {title}')
            _hijack_weight_prune(m_, value)

def hijack_act_prune_modules(prune_dict, name, m):
    value = prune_dict.get(name, None)
    if value is not None:
        print(f'Mixed act command {name}')
    for title, m_ in m.named_modules():
        if hasattr(m_, 'activation_pruner'):
            if value is not None:
                print(f'Mixed act pruned {title}')
            _hijack_act_prune(m_, value)


def set_act_prune_axis_and_groups(module, axis, n_groups, permute=False):
    if hasattr(module, 'activation_pruner'):
        module = module.activation_pruner

    module.axis = axis
    module.pruner.axis = axis
    module.range_estimator.axis = axis

    module.n_groups = n_groups
    module.range_estimator.n_groups = n_groups

    if permute:
        module.range_estimator.per_group_range_estimation = True

    return module
