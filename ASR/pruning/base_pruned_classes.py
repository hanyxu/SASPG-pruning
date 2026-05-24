# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn

# from pruning_manager import PruningManager
# from pruning.pruners import PMethods

# from pruning_manager_direct import PruningManager
# from pruning.pruners_direct import PMethods
# from pruning_manager_direct_8_6_4_split_init import PruningManager
from pruning.pruning_manager_direct import PruningManager
from pruning.pruners_direct import PMethods
# from pruning.pruners_switch import PMethods_switch

from pruning.range_estimators import RangeEstimators
import torch

def _set_layer_learn_ranges(layer):
    if isinstance(layer, PruningManager):
        if layer.pruner.pruner.is_initialized:
            layer.learn_ranges()


def _set_layer_fix_ranges(layer):
    if isinstance(layer, PruningManager):
        if layer.pruner.pruner.is_initialized: # 永远都是is_initialized False
            layer.fix_ranges()


def _set_layer_estimate_ranges(layer):
    if isinstance(layer, PruningManager):
        if layer.pruner.is_initialized:
            layer.estimate_ranges()


def _set_layer_estimate_ranges_train(layer):
    if isinstance(layer, PruningManager):
        if layer.pruner.is_initialized:
            layer.estimate_ranges_train()


class PrunedModule(nn.Module): # replace the function of Hijacker which does not have __init__
    """
    Parent class for a pruned module. It adds the basic functionality of switching the module
    between pruned and full precision mode. It also defines the cached parameters and handles
    the reset of the cache properly.
    """
    def __init__(self, 
        *args,
        method = PMethods.asymmetric_uniform,
        num_acts = 1,
        act_method = None,
        n_bits = 8,
        n_bits_act = None,
        per_channel_weights = False,
        act_momentum = 0.1,
        per_channel_acts = False,
        percentile = None,
        weight_range_method = RangeEstimators.current_minmax,
        weight_range_options = None,
        act_range_method = RangeEstimators.running_minmax,
        act_range_options = None,
        scale_domain = "linear",
        gating_method = "l0",
        gate_init_dict = None,
        learned_scale = False,
        clip_input = False,
        checkpointing = False,
        include_pruning = False,
        channel_pruning = False,
        value_0_5 = 0.0,
        value_0_75 = 0.0,
        value_0_25 = 0.0,
        value_0_125 = 0.0,
        value_1 = 0.0,
        value_0_1 = 0.0,
        value_0_075 = 0.0,
        reg_type = "const",
        prune_only = False,
        fixed_bit_dict = None,
        fix_prob = None,
        is_out_proj = False,
        is_load = False,
        lmb = 0,
        single_bit = 0,
        lmb_dis = 0.0,
        weight_prune = True,
        act_prune = False,
        max_bit = 3.9,
        min_bit = 3.8,
        max_prune_ratio = 0.15,
        min_prune_ratio = 0.148,
        mag_prune = False,
        hand_ratio = 0.15,
        hard = False,
        only_size_hard = True,
        decay_tau = True,
        prune_2_4 = False,
        is_arc_prune = False,
        save_path = None,
        thre_model_path = None,
        model_path = None,
        total_steps = None,
        eta_max = None,
        eta_min = None,
        prune_dict = {"only_t": 1},
        **kwargs
        ):
    
        
        _DEFAULT_CONFIG = {
            "method": PMethods.asymmetric_uniform,
            "num_acts": 1,
            "act_method": None,
            "n_bits": 8,
            "n_bits_act": None,
            "per_channel_weights": False,
            "act_momentum": 0.1,
            "per_channel_acts": False,
            "percentile": None,
            "weight_range_method": RangeEstimators.current_minmax,
            "weight_range_options": None,
            "act_range_method": RangeEstimators.running_minmax,
            "act_range_options": None,
            "scale_domain": "linear",
            "gating_method": "l0",
            "gate_init_dict": None,
            "learned_scale": False,
            "clip_input": False,
            "checkpointing": False,
            "include_pruning": False,
            "channel_pruning": False,
            "value_0_5": 0.0,
            "value_0_75": 0.0,
            "value_0_25": 0.0,
            "value_0_125": 0.0,
            "value_1": 0.0,
            "value_0_1": 0.0,
            "value_0_075": 0.0,
            "reg_type": "const",
            "prune_only": False,
            "fixed_bit_dict": None,
            "fix_prob": None,
            "is_out_proj": False,
            "is_load": False,
            'lmb':0, 
            'single_bit': 0, 
            'lmb_dis': 0.0, 
            'weight_prune': True, 
            'act_prune': False, 
            'max_bit': 3.9, 
            'min_bit': 3.8, 
            'max_prune_ratio': 0.15, 
            'min_prune_ratio': 0.148, 
            'mag_prune': False, 
            'hand_ratio': 0.15, 
            'hard': False, 
            'only_size_hard': True, 
            'decay_tau': True, 
            'prune_2_4': False, 
            'is_arc_prune': False, 
            'save_path': None, 
            'model_path':None,
            'thre_model_path':None,
            'eta_max': None,
            'eta_min': None,
            'prune_dict': {'only_t': 1}}
        

        # SASPG/NASP/MAG str flags live on prune_params for channel models; every Linear
        # still receives **prune_params via autoprune_utils. They must not reach nn.Linear.
        kwargs.pop("mag_structural", None)
        kwargs.pop("nasp_ladder", None)

        super().__init__(*args, **kwargs)
        
            
        self.method = method
        self.num_acts = num_acts
        self.act_method = act_method
        self.n_bits = n_bits
        self.n_bits_act = n_bits_act
        self.per_channel_weights = per_channel_weights
        self.act_momentum = act_momentum
        self.per_channel_acts = per_channel_acts
        self.percentile = percentile
        self.weight_range_method = weight_range_method
        self.weight_range_options = weight_range_options if weight_range_options else {}
        self.act_range_method = act_range_method
        self.act_range_options = act_range_options if act_range_options else {}
        self.scale_domain = scale_domain
        self.gating_method = gating_method
        self.gate_init_dict = gate_init_dict
        self.learned_scale = learned_scale
        self.clip_input = clip_input
        self.checkpointing = checkpointing
        self.include_pruning = include_pruning
        self.channel_pruning = channel_pruning
        self.value_0_5 = value_0_5
        self.value_0_75 = value_0_75
        self.value_0_25 = value_0_25
        self.value_0_125 = value_0_125
        self.value_1 = value_1
        self.value_0_1 = value_0_1
        self.value_0_075 = value_0_075
        self.reg_type = reg_type
        self.prune_only = prune_only
        self.fixed_bit_dict = fixed_bit_dict
        self.fix_prob = fix_prob
        self.is_out_proj = is_out_proj
        self.is_load = is_load
        self.lmb = lmb
        self.single_bit = single_bit
        self.lmb_dis = lmb_dis
        self.weight_prune = weight_prune
        self.act_prune = act_prune
        self.max_bit = max_bit
        self.min_bit = min_bit
        self.max_prune_ratio = max_prune_ratio
        self.min_prune_ratio = min_prune_ratio
        self.mag_prune = mag_prune
        self.hand_ratio = hand_ratio
        self.hard = hard
        self.only_size_hard = only_size_hard
        self.decay_tau = decay_tau
        self.prune_2_4 = prune_2_4
        self.is_arc_prune = is_arc_prune
        self.save_path = save_path
        self.thre_model_path = thre_model_path
        self.model_path = model_path
        self.prune_dict = prune_dict
        self.total_steps = total_steps
        self.eta_max = eta_max
        self.eta_min = eta_min
        
        self.cached_params = None
        self._caching = False

        self.prune_params = None
        self._prune_w = False
        self._prune_a = False
        
        
        self._return_dict = {
            'mask': False,
            'unmask': False,
            'fp': False,
            'q': False,
            'q8': False,
            'q7': False,
            'q6': False,
            'q5': False,
            'q4': False,
            'q3': False,
            'q2': False,
            'q1': False,
        }
        
        
    @property
    def caching(self):
        return self._caching

    @caching.setter
    def caching(self, value: bool):
        self._caching = value
        if not value:
            self.cached_params = None

    def pruned_weights(self):
        self.cached_params = None
        self._prune_w = True


    def full_precision_weights(self):
        self.cached_params = None
        self._prune_w = False

    def pruned_acts(self):
        self._prune_a = True

    def full_precision_acts(self):
        self._prune_a = False

    def pruned(self):
        self.pruned_weights()
        self.pruned_acts()

    def full_precision(self):
        self.full_precision_weights()
        self.full_precision_acts()

    def update_return(self, key):
        for k in self._return_dict.keys():
            self._return_dict[k] = False
        self._return_dict[key] = True

    def return_mask(self):
        self.update_return('mask')
    
    def return_unmask(self):
        self.update_return('unmask')
        
    def return_fp(self):
        self.update_return('fp')

    def return_q(self):
        self.update_return('q')

    def return_q8(self):
        self.update_return('q8')

    def return_q4(self):
        self.update_return('q4')
        
    def return_q5(self):
        self.update_return('q5')

    def return_q3(self):
        self.update_return('q3')
        
    def return_q2(self):
        self.update_return('q2')
        
    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.cached_params = None
        return self

    def _apply(self, *args, **kwargs):
        self.cached_params = None
        return super(PrunedModule, self)._apply(*args, **kwargs)

    def extra_repr(self):
        prune_state = 'weight_prune={}, act_prune={}'.format(self._prune_w, self._prune_a)
        parent_repr = super().extra_repr()
        return '{},\n{}'.format(parent_repr, prune_state) if parent_repr else prune_state


class PrunedActivation(PrunedModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        act_prune_params = dict(n_bits=self.n_bits_act, scale_domain=self.scale_domain)
        
        
        self.activation_pruner = PruningManager(
            qmethod=self.act_method,
            init=self.act_range_method,
            per_channel=self.per_channel_acts,
            prune_params=act_prune_params,
            init_params=self.act_range_options,
            gating_method=self.gating_method,
            gate_init_dict=self.gate_init_dict,
            fixed_bit_dict=self.fixed_bit_dict,
            act_prune=True,
            checkpointing=self.checkpointing,
            reg_type=self.reg_type,
        )
        
    def prune_activations(self, x):
        if self._prune_a:
            return self.activation_pruner(x)
        else:
            return x

    def forward(self, x):
        return self.prune_activations(x)


class FP32Acts(nn.Module):
    def forward(self, x):
        return x

    def reset_ranges(self):
        pass
