# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from enum import Enum

from torch import nn

from pruning.pruners_direct import AsymmetricUniformPruner, SymmetricUniformPruner, PMethods, PrunerNotInitializedError
# from pruning.pruners_switch import PMethods_switch
from pruning.range_estimators import RangeEstimators

import torch
class Qstates(Enum):
    estimate_ranges = 0  # ranges are updated in eval and train mode
    fix_ranges = 1  # pruning ranges are fixed for train and eval
    learn_ranges = 2  # pruning params are nn.Parameters
    estimate_ranges_train = 3  # pruning ranges are updated during train and fixed for eval


class PruningManager(nn.Module):
    """Implementation of Pruning and Pruning Range Estimation

    Parameters
    ----------
    n_bits: int
        Number of bits for the pruning.
    qmethod: PMethods member (Enum)
        The pruning scheme to use, e.g. symmetric_uniform, asymmetric_uniform,
        qmn_uniform etc.
    init: RangeEstimators member (Enum)
        Initialization method for the grid from
    per_channel: bool
        If true, will use a separate pruning grid for each kernel/channle.
    x_min: float or PyTorch Tensor
        The minimum value which needs to be represented.
    x_max: float or PyTorch Tensor
        The maximum value which needs to be represented.
    """
    def __init__(self, 
                qmethod=PMethods.symmetric_uniform,
                init=RangeEstimators.current_minmax,
                num_acts=16,
                gating_method="pg",
                gate_init_dict=None,
                act_prune=False,
                checkpointing=False,
                include_pruning=False,
                channel_pruning=False,
                value_0_5 = 0.,
                value_0_75 = 0.,
                value_0_25 = 0.,
                value_0_125 = 0.,
                value_1 = 0.,
                value_0_1 = 0.,
                value_0_075 = 0.,
                prune_only=False,
                fixed_bit_dict=None,
                reg_type="const",
                per_channel=False,
                axis=None,
                n_groups=None,
                x_min=None,
                x_max=None,
                prune_params=None,
                init_params=None,
                gate_dict=None,
                return_bit_dict=None,
                fix_prob=None,
                is_out_proj=False):
        super().__init__()
        self.state = Qstates.estimate_ranges
        # self.precision_level = prune_params['n_bits'] or prune_params['n_bits_act']
        
        self.qmethod = qmethod
        self.init = init
        self.per_channel = per_channel
        self.axis = axis
        self.n_groups = n_groups
        self.prune_params = prune_params if prune_params else {}
        
        self.init_params = init_params if init_params else {}
        
        # self.range_estimator = None
        # self.range_estimator_6 = None
        # self.range_estimator_4 = None
        # self.range_estimator_3 = None
        # self.range_estimator_2 = None
        
        self.x_min = x_min
        self.x_max = x_max
        self.x = None
        
        self.is_feature_ext = False
        # self.num_acts = num_acts
        
        gate_dict = {} if gate_dict is None else gate_dict
        _pruner_extra = dict(prune_params) if prune_params else {}
        for _k in (
            "nasp_ladder", "model_path", "thre_model_path", "save_path",
            "weight_range_method", "weight_range_options", "prune_setup", "method",
            "lmb", "lmb_dis", "single_bit", "weight_prune", "act_prune", "max_bit", "min_bit",
            "max_prune_ratio", "min_prune_ratio", "mag_prune", "hand_ratio", "hard",
            "only_size_hard", "decay_tau", "prune_2_4", "is_arc_prune", "per_channel_weights",
            "percentile", "gating_method", "gate_init_dict", "include_pruning", "channel_pruning",
            "prune_only", "reg_type", "fixed_bit_dict", "total_steps", "eta_max", "eta_min",
            "value_1", "value_0_75", "value_0_5", "value_0_25", "value_0_125", "value_0_1", "value_0_075",
        ):
            _pruner_extra.pop(_k, None)
        # import pdb;pdb.set_trace()
        self.pruner = self.qmethod.cls(
            num_acts=num_acts,
            method='bayesian_bits',
            gating_method=gating_method,
            gate_init_dict=gate_init_dict,
            act_prune=act_prune,
            checkpointing=checkpointing,
            include_pruning=include_pruning,
            channel_pruning=channel_pruning,
            value_0_5 = value_0_5,
            value_0_75 = value_0_75,
            value_0_25 = value_0_25,
            value_0_125 = value_0_125,
            value_1 = value_1,
            value_0_1 = value_0_1,
            value_0_075 = value_0_075,
            prune_only=prune_only,
            fixed_bit_dict=fixed_bit_dict,
            reg_type=reg_type,
            gate_dict=gate_dict,
            return_bit_dict=return_bit_dict,
            fix_prob=fix_prob,
            is_out_proj=is_out_proj,
            **_pruner_extra)

        # define range estimation method for pruner initialisation
        if x_min is not None and x_max is not None:
            self.set_prune_range(x_min, x_max)
            self.state = Qstates.fix_ranges 
        else:
            assert ('percentile' not in self.init_params) or (not self.init_params['percentile'])
            self.range_estimator = self.init.cls(
                    per_channel=self.per_channel,
                    pruner=self.pruner,
                    axis=self.axis,
                    n_groups=self.n_groups,
                    **self.init_params
                )
            
            self.range_estimator_4 = self.init.cls(
                    per_channel=self.per_channel,
                    pruner=self.pruner,
                    axis=self.axis,
                    n_groups=self.n_groups,
                    **self.init_params

                )
            # self.range_estimator_5 = self.init.cls(
            #         per_channel=self.per_channel,
            #         pruner=self.pruner,
            #         axis=self.axis,
            #         n_groups=self.n_groups,
            #         **self.init_params
            #     )
            self.range_estimator_2 = self.init.cls(
                    per_channel=self.per_channel,
                    pruner=self.pruner,
                    axis=self.axis,
                    n_groups=self.n_groups,
                    **self.init_params
                )
            # self.range_estimator_3 = self.init.cls(
            #         per_channel=self.per_channel,
            #         pruner=self.pruner,
            #         axis=self.axis,
            #         n_groups=self.n_groups,
            #         **self.init_params
            #     )
            

    @property
    def n_bits(self):
        return self.pruner.pruner.n_bits
    
    def get_full_class_name(self):
        return f"{self.__module__}.{self.__class__.__name__}"
    
    def get_gate_loss(self):
        regularizer = 0.0
        if hasattr(self.pruner, 'pruner'):
            regularizer += self.pruner.pruner.regularizer()
        return regularizer

    def get_gate_loss_direct(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
            # temp_loss = self.pruner.pruner.regularizer_size()
            # print('self.pruner.pruner.regularizer_size()',self.pruner.pruner.regularizer_size())
            regularizer += self.pruner.pruner.regularizer_size()
        return regularizer
    
    def get_gate_loss_prune(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
            # temp_loss = self.pruner.pruner.regularizer_size()
            # print('self.pruner.pruner.regularizer_size()',self.pruner.pruner.regularizer_size())
            regularizer += self.pruner.pruner.regularizer_size_prune()
        return regularizer
    
    def get_gate_loss_prune_channel_gate(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
            # temp_loss = self.pruner.pruner.regularizer_size()
            # print('self.pruner.pruner.regularizer_size()',self.pruner.pruner.regularizer_size())
            regularizer += self.pruner.pruner.regularizer_size_prune_channel_gate()
        return regularizer
    
    def get_gate_loss_prune_channel(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
            # temp_loss = self.pruner.pruner.regularizer_size()
            # print('self.pruner.pruner.regularizer_size()',self.pruner.pruner.regularizer_size())
            regularizer += self.pruner.pruner.regularizer_size_prune_channel()
        return regularizer

    def get_gate_loss_prune_channel10(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
            # temp_loss = self.pruner.pruner.regularizer_size()
            # print('self.pruner.pruner.regularizer_size()',self.pruner.pruner.regularizer_size())
            regularizer += self.pruner.pruner.regularizer_size_prune_channel10()
        return regularizer
    
    def fix_threshold(self):
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner'):
                self.pruner.pruner.fix_threshold()

    
    def get_gate_loss_distil(self):
        # print(self.pruner.pruner.symmetric)
        regularizer = 0.
        # print('get_gate_loss_distil...')
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.regularizer_distil()
        return regularizer
    
    def get_gate_loss_disweight(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.regularizer_disweight()
        return regularizer
    
    def get_mse_weight(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.regularizer_mse_pruned()
        return regularizer
    
    def get_gate_loss_disweightout(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.regularizer_disweightout()
        return regularizer

    def get_exact_size(self):
        regularizer = 0.0
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.get_exact_size()
        return regularizer
    
    def get_exact_size_prune(self):
        regularizer = 0.0
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.get_exact_size_prune()
        return regularizer

    def get_gate_loss_disLinear(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.regularizer_disLinear()
        return regularizer


    def get_gate_loss_KL(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner'):
                regularizer += self.pruner.pruner.regularizer_KL()
        return regularizer
    
    def get_gate_loss_L1(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.use_L1 is True:
                regularizer += self.pruner.pruner.regularizer_L1()
        return regularizer
    
    def get_gate_loss_cos(self):
        regularizer = 0.0
        # import pdb;pdb.set_trace()
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.use_cos is True:
                regularizer += self.pruner.pruner.regularizer_cos()
        return regularizer

    def get_mixed_prec(self):
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
                mixed_bits =  self.pruner.pruner.output_mixed_prec()
                return mixed_bits
            
    def get_mixed_prec_fix(self):
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
                mixed_bits =  self.pruner.pruner.output_mixed_prec_fix()
                return mixed_bits
            
    def get_mixed_sparsity(self):
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
                mixed_bits =  self.pruner.pruner.output_mixed_sparsity()
                return mixed_bits
            
    def get_mixed_sparsity_fix(self):
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
                mixed_bits =  self.pruner.pruner.output_mixed_sparsity_fix()
                return mixed_bits
            
    def get_mixed_thre_fix(self):
        if hasattr(self.pruner, 'pruner') and self.pruner.pruner.symmetric:
                mixed_bits =  self.pruner.pruner.output_mixed_thre_fix()
                return mixed_bits
    
    def set_precision_level_trunc(self, precision):
        # print(self.pruner.is_initialized)
        if hasattr(self, 'pruner') and self.pruner.is_initialized and self.pruner.symmetric:
            # import pdb;pdb.set_trace()
            if precision == 4:
                self.pruner.extra_bits = 4
                # print(self.pruner.n_bits ,self.state)
                # assert self.state == Qstates.learn_ranges
                # self.learn_ranges()
            if precision == 8: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.extra_bits = 8
                # print(self.pruner.n_bits ,self.state)
                # assert self.state == Qstates.learn_ranges
                # self.learn_ranges()
            if precision == 5: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.extra_bits = 5
                # print(self.pruner.n_bits ,self.state)
                # assert self.state == Qstates.learn_ranges
                # self.learn_ranges()
            if precision == 6: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.extra_bits = 6
                # print(self.pruner.n_bits ,self.state)
                # assert self.state == Qstates.learn_ranges
                # self.learn_ranges()
            if precision == 7: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.extra_bits = 7
                # print(self.pruner.n_bits ,self.state)
                # assert self.state == Qstates.learn_ranges
                # self.learn_ranges()
            if precision == 2: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.extra_bits = 2
                # print(self.pruner.n_bits ,self.state)
                # assert self.state == Qstates.learn_ranges
                # self.learn_ranges()

    def set_delta(self, weight_dict): # not used
        if hasattr(self, 'pruner'):
            self.pruner.pruner._delta
            self.pruner.pruner._delta_4bit
            self.pruner.pruner._delta_2bit
            import pdb;pdb.set_trace()
        
    def set_precision_level_direct(self, precision):
        # print(self.pruner.is_initialized)
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized:
        if hasattr(self, 'pruner'):
            # import pdb;pdb.set_trace()
            if precision == 4:
                self.pruner.pruner.n_bits = 4
             
            if precision == 8: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.pruner.n_bits = 8
              
            if precision == 5: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.pruner.n_bits = 5
              
            if precision == 6: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.pruner.n_bits = 6
               
            if precision == 7: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.pruner.n_bits = 7
               
            if precision == 2: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.pruner.n_bits = 2
            
            if precision == 3: # and hasattr(self.pruner,'extra_bits'):
                self.pruner.pruner.n_bits = 3
                
    def set_total_steps(self, total_steps):
        # print(self.pruner.is_initialized)
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized:
        """if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric:"""
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.total_steps == 0
            self.pruner.pruner.total_steps = total_steps
            
    def set_fix_prob(self):
        # print(self.pruner.is_initialized)
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized:
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric: # for pruning
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.fix_prob == False
            self.pruner.pruner.fix_prob = True
            
    def set_mag_prune(self):
        # print(self.pruner.is_initialized)
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized:
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric: # for pruning
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.mag_prune == False
            self.pruner.pruner.mag_prune = True
            
    def set_hand_ratio(self, hand_ratio):
        # print(self.pruner.is_initialized)
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized:
        # if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric: # for pruning
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.hand_ratio == False
            self.pruner.pruner.hand_ratio = hand_ratio
     
            
    def set_gumbel_hard(self):
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.is_hard == False
            self.pruner.pruner.is_hard = True
            
    def set_prob_prune(self, prob_list):
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.prob_1 == 0.
            assert self.pruner.pruner.prob_05 == 0.
            assert self.pruner.pruner.prob_025 == 0.
            assert self.pruner.pruner.prob_0125 == 0.
            
            self.pruner.pruner.prob_1 = prob_list[0]
            self.pruner.pruner.prob_05 = prob_list[1]
            self.pruner.pruner.prob_025 = prob_list[2]
            self.pruner.pruner.prob_0125 = prob_list[3]

    """def set_only_size_hard(self): # for pruning 
        if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.only_size_hard == False
            self.pruner.pruner.only_size_hard = True"""
            
    def set_only_size_hard(self): # for pruning
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.only_size_hard == False
            self.pruner.pruner.only_size_hard = True
            
    def set_decay_tau(self): # for pruning
        if hasattr(self, 'pruner') and self.pruner.pruner.symmetric:
            assert self.pruner.pruner.decay_tau == False
            self.pruner.pruner.decay_tau = True

    def set_prune_2_4(self):
        if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.prune_2_4 == False
            self.pruner.pruner.prune_2_4 = True
    
    def set_prune_4_8(self):
        if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.prune_4_8 == False
            assert self.pruner.pruner.prune_2_4 == False
            self.pruner.pruner.prune_4_8 = True
            
    def set_is_arc_prune(self):
        if hasattr(self, 'pruner') and self.pruner.pruner.is_initialized and self.pruner.pruner.symmetric:
            # import pdb;pdb.set_trace()
            assert self.pruner.pruner.is_arc_prune == False
            self.pruner.pruner.is_arc_prune = True
                
        
    def estimate_ranges(self):
        self.state = Qstates.estimate_ranges

    def fix_ranges(self):
        if self.pruner.pruner.is_initialized:
            self.state = Qstates.fix_ranges
        else:
            raise PrunerNotInitializedError()

    def learn_ranges(self):
        """
        if self.pruner.pruner.x_shape_0 == 512 or self.pruner.pruner.x_shape_1 == 512:
            self.pruner.pruner.make_range_trainable_8bit()
        """
            
        self.pruner.pruner.make_range_trainable()
        # print('.make_range_trainable_8bit>>>')
        # self.pruner.pruner.make_range_trainable_4bit()
        # print('.make_range_trainable_4bit>>>')
        # self.pruner.pruner.make_range_trainable_3bit()
        # print('.make_range_trainable_3bit>>>')
        # # self.pruner.make_range_trainable_6bit()
        # # print('.make_range_trainable_6bit>>>')
        # self.pruner.pruner.make_range_trainable_2bit()
        # print('.make_range_trainable_2bit>>>')
        self.state = Qstates.learn_ranges
    

    def estimate_ranges_train(self):
        self.state = Qstates.estimate_ranges_train

    def reset_ranges(self):
        self.range_estimator.reset()
        self.pruner.reset()
        self.estimate_ranges()

    def forward(self, x):
        """
        if self.range_estimator.per_group_range_estimation:
            self.range_estimator(x)
            return x
        """
        if isinstance(x, list):
            weight_and_x = x
            x = x[0]
        else:
            weight_and_x = x

        # print('self.state:', self.state) 
        """only prune here means self._delta is not initialized"""
        # if self.state == Qstates.estimate_ranges or (self.state == Qstates.estimate_ranges_train and self.training):

        #     import pdb;pdb.set_trace()
        #     if self.state == Qstates.estimate_ranges:
                
        #         # if self.is_feature_ext:
        #         #     cur_xmin, cur_xmax = self.range_estimator(x)
        #         #     self.n_bits = 8
        #         # assert self.n_bits == 8

        #         if self.n_bits == 8:
        #             assert type(self.range_estimator == self.range_estimator_4)
                   
        #             cur_xmin, cur_xmax = self.range_estimator(x)
                    
        #         elif self.n_bits == 6:
        #             assert isinstance(self.range_estimator_6, RangeEstimators.MSE.cls)
        #             cur_xmin, cur_xmax = self.range_estimator_6(x)
                    
        #         elif self.n_bits == 4:
        #             assert type(self.range_estimator == self.range_estimator_4)
                   
        #             cur_xmin, cur_xmax = self.range_estimator_4(x)
                    
        #         elif self.n_bits == 5:
        #             assert type(self.range_estimator == self.range_estimator_5)
                   
        #             cur_xmin, cur_xmax = self.range_estimator_5(x)
                    
        #         elif self.n_bits == 2:
        #             assert type(self.range_estimator == self.range_estimator_2)
                   
        #             cur_xmin, cur_xmax = self.range_estimator_2(x)

        #         elif self.n_bits == 3:
        #             assert type(self.range_estimator == self.range_estimator_3)
                   
        #             cur_xmin, cur_xmax = self.range_estimator_3(x)
              
        #         print('set_prune_range:',self.n_bits)
                
    
        #         self.set_prune_range(cur_xmin, cur_xmax, self.n_bits)

                # if self.n_bits == 2:
                #     print('self.pruner.pruner._delta_2bit',self.pruner.pruner._delta_2bit)
        # print(self.get_full_class_name())
        # import pdb;pdb.set_trace()
        return self.pruner(weight_and_x)

    def set_prune_range(self, x_min, x_max, signal_bits):
        self.pruner.pruner.set_prune_range(x_min, x_max, self.n_bits)

    def extra_repr(self):
        return 'state={}'.format(self.state.name)
