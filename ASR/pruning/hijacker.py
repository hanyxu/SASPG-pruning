# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy

import torch
from torch import nn
import torch.nn.functional as F

from pruning.base_pruned_classes import PrunedModule
# from pruning_manager_direct import PruningManager
from pruning.base_pruned_classes import FP32Acts
# from pruning_manager import PruningManager

# from pruning_manager_direct_8_6_4_split_init import PruningManager
from pruning_manager_direct import PruningManager

from pruning.range_estimators import RangeEstimators
from pruning.utils import to_numpy


activations_list = [nn.ReLU, nn.ReLU6, nn.Hardtanh, nn.Sigmoid, nn.Tanh, nn.PReLU, nn.GELU]

class PruningHijacker(PrunedModule):
    """Mixin class that 'hijacks' the forward pass in a module to perform pruning and
    depruning on the weights and output distributions.

    Usage:
    To make a pruned nn.Linear layer:
    ```
    >>> class PruneLinear(PruningHijacker, nn.Linear):
    ...     pass
    ```

    It is vital that QSchemeForwardHijacker is the first parent class, and that the second parent
    class derives from nn.Module, otherwise it will not be reached by a super(., .) call.

    NB: this implementation (for now) assumes that there will always be some training involved,
    e.g. to estimate the activation ranges.
    """
    def __init__(self, *args, activation: nn.Module = None, **kwargs):
        super().__init__(*args, **kwargs)
        if activation:
            assert isinstance(activation, tuple(activations_list))
                    
        self.activation_function = copy.deepcopy(activation) if activation else None

        self.layer_name = None
        
        # tensor = torch.tensor(1e-5).to("cuda" if self.cuda else "cpu")
       
        # setattr(self, f'threshold_prune_channel', torch.nn.Parameter(tensor))
        self.threshold_prune_channel = torch.nn.Parameter(torch.tensor(30.),requires_grad=True).to('cuda')

        weight_prune_params = dict(n_bits=self.n_bits, scale_domain=self.scale_domain)
        act_prune_params = dict(n_bits=self.n_bits_act, scale_domain=self.scale_domain)
        
        # self.threshold_list = [self.gate_init_dict['q']] * 3
        """
        self.activation_pruner = PruningManager(
            num_acts=self.num_acts,
            qmethod=self.act_method,
            init=self.act_range_method,
            per_channel=self.per_channel_acts,
            prune_params=act_prune_params,
            init_params=self.act_range_options,
            gating_method=self.gating_method,
            gate_init=self.gate_init,
            gate_8_init=self.gate_8_init,
            gate_4_init=self.gate_4_init,
            gate_2_init=self.gate_2_init,
            include_pruning=self.include_pruning,
            fixed_8bit=(self.fixed_8bit or self.fixed_48bit),
            act_prune=True,
            checkpointing=self.checkpointing,
            reg_type=self.reg_type,
        )
        """
            
        if self.weight_range_method == RangeEstimators.current_minmax: 
            # if current_minmax then weight_range_options cotaining golden section/grid will not bed used.
            weight_init_params = dict(percentile=self.percentile)
        else:
            weight_init_params = self.weight_range_options
    
        """self.weight_pruneizer = PruningManager(
            num_acts=self.num_acts,
            qmethod=self.method,
            init=self.weight_range_method,
            per_channel=self.per_channel_weights,
            prune_params=weight_prune_params,
            init_params=weight_init_params,
            gating_method=self.gating_method,
            gate_init_dict=self.gate_init_dict,
            checkpointing=self.checkpointing,
            include_pruning=self.include_pruning,
            reg_type=self.reg_type,
            prune_only=self.prune_only,
            fixed_bit_dict=self.fixed_bit_dict,
            return_bit_dict=self._return_dict,
            fix_prob=self.fix_prob,
            is_out_proj=self.is_out_proj,
        )""" # if used then get_size_loss will double time caculated
        
        self.weight_pruneizer_saspg = PruningManager(
            num_acts=self.num_acts,
            qmethod=self.method,
            init=self.weight_range_method,
            per_channel=self.per_channel_weights,
            prune_params=weight_prune_params,
            init_params=weight_init_params,
            gating_method=self.gating_method,
            gate_init_dict=self.gate_init_dict,
            checkpointing=self.checkpointing,
            include_pruning=self.include_pruning,
            channel_pruning=self.channel_pruning,
            value_0_5 = self.value_0_5,
            value_0_75 = self.value_0_75,
            value_0_25 = self.value_0_25,
            value_0_125 = self.value_0_125,
            value_0_075 = self.value_0_075,
            value_0_1 = self.value_0_1,
            value_1 = self.value_1,
            reg_type=self.reg_type,
            prune_only=self.prune_only,
            fixed_bit_dict=self.fixed_bit_dict,
            return_bit_dict=self._return_dict,
            fix_prob=self.fix_prob,
            is_out_proj=self.is_out_proj,
        )
        
        self.activation_save_target = None
        self.activation_save_name = None
        self.param_list = []
        
        
        

    def forward_orig(self, x, offsets=None):
            
        weight, weight_q, bias, gate_2, gate_4, gate_8 = self.get_params()
      
        res_q = self.run_forward(x, weight_q, bias, offsets=offsets)
        res_q = self.prune_activations(res_q, gate_2, gate_4, gate_8)
        
        if 'weight' not in self.reg_type:
            res_fp = self.run_forward(x, weight, bias, offsets=offsets)
        elif 'weight' in self.reg_type:
            res_8 = self.run_forward(x, weight, bias, offsets=offsets)
        
        # if self.training:
        #     import pdb;pdb.set_trace()
        if self.training and (self.reg_type == 'distilctc' or self.reg_type == 'distil') and self.weight_pruneizer.pruner.pruner.use_distil is True: # used
            self.weight_pruneizer.pruner.pruner.distil_loss = nn.L1Loss()(res_q, res_fp.detach())
            # print('self.weight_pruneizer.pruner.pruner.distil_loss', res_q.shape, self.weight_pruneizer.pruner.pruner.distil_loss)
        
        if self.training and self.reg_type == 'disweightout': # for evey pruner we use ditill loss on [output]
            self.weight_pruneizer.pruner.pruner.disweightout_loss = nn.L1Loss()(res_q, res_8.detach())
            
        elif self.training and self.reg_type == 'disweight': # for evey pruner we use ditill loss on [weight]
            res_8 = self.run_forward(x, weight, bias, offsets=offsets)
            
        if self._return_dict['q']:
            return res_q
        
        elif self._return_fp:
            if self.reg_type == 'disweight' or self.reg_type == 'disweightout':
                return res_8
            else:
                return res_fp
            
    def forward(self, x, offsets=None):
            
        # weight, bias, weight_list, gate_dict = self.get_params()
        
        weight, bias = self.get_params_saspg()
        # weight, bias = self.get_params_channel_vec()

        if self._prune_w:
            
            res_fp = self.run_forward(x, weight, bias, offsets=offsets)
            # print('weight',weight.shape)
            # no activation pruned? 2024.6.13

            return res_fp
        
            # print(weight.shape) # first to o is check the size of CNN
            if weight.numel() != 3072*768 and weight.numel() != 768*768 and weight.numel() != 1024*4096 and weight.numel() != 1024*1024 and self.training:
                # print(weight.shape)
                # import pdb;pdb.set_trace()
                # print(len(list(weight_list[0].unique())))
                res_8 = self.run_forward(x, weight_list[0], bias, offsets=offsets)
                res_8 = self.prune_activations(res_8, gate_dict)
                
                return res_8
            
            
            if self._return_dict['q']:
                res_q = self.run_forward(x, weight_list[0], bias, offsets=offsets)
                res_q = self.prune_activations(res_q, gate_dict)
                
                return res_q
            
            elif self._return_dict['q3']:
                res_3 = self.run_forward(x, weight_list[3], bias, offsets=offsets)
                res_3 = self.prune_activations(res_3, gate_dict)
                
                return res_3
                
            elif self._return_dict['q8']:
                res_8 = self.run_forward(x, weight_list[8], bias, offsets=offsets)
                res_8 = self.prune_activations(res_8, gate_dict)
                
                return res_8
            
            elif self._return_dict['q4']:
                res_4 = self.run_forward(x, weight_list[4], bias, offsets=offsets)
                res_4 = self.prune_activations(res_4, gate_dict)
                
                return res_4
            
            elif self._return_dict['q5']:
                res_5 = self.run_forward(x, weight_list[5], bias, offsets=offsets)
                res_5 = self.prune_activations(res_5, gate_dict)
                
                return res_5
            
            elif self._return_dict['q2']:
                res_2 = self.run_forward(x, weight_list[2], bias, offsets=offsets)
                res_2 = self.prune_activations(res_2, gate_dict) 

                return res_2
            
            else:
                return res_fp
        else:
            # import pdb;pdb.set_trace()
            if self._return_dict['mask']:
                res_fp = self.run_forward(x, weight, bias, offsets=offsets)
                return res_fp
            
            elif self._return_dict['unmask']:
                res_unmask = self.run_forward(x, weight_unmask, bias, offsets=offsets)
                return res_unmask
        
            else:
                return self.run_forward(x, weight, bias, offsets=offsets)
        
    def get_params_v2(self):
        # if not self.training and self.cached_params:
        #     return self.cached_params

        weight, bias = self.get_weight_bias()

        if self._prune_w:
            weight_q, weight_8, weight_4, weight_2 = self.weight_pruneizer(weight)
            gate_2, gate_4, gate_8 = self.weight_pruneizer.pruner.pruner.return_gates()
            
            if self.training and (self.reg_type == 'disweightout' or self.reg_type == 'disweight'):
                self.weight_pruneizer.pruner.pruner.fixed_8bit = True
                weight_8 = self.weight_pruneizer(weight)
                self.weight_pruneizer.pruner.pruner.fixed_8bit = False
            
                return weight_8, weight_q, bias, gate_2, gate_4, gate_8
            
            # elif self.training and (self.reg_type == 'distilctc'):
            #     if self._return_dict['q2']:
            #         self.weight_pruneizer.pruner.pruner.fixed_2bit = True
            #         weight_2 = self.weight_pruneizer(weight)
            #         self.weight_pruneizer.pruner.pruner.fixed_2bit = False
                
            #     if self._return_dict['q4']:
            #         self.weight_pruneizer.pruner.pruner.fixed_4bit = True
            #         weight_4 = self.weight_pruneizer(weight)
            #         self.weight_pruneizer.pruner.pruner.fixed_4bit = False
                
            #     if self._return_dict['q8']:
            #         self.weight_pruneizer.pruner.pruner.fixed_8bit = True
            #         weight_8 = self.weight_pruneizer(weight)
            #         self.weight_pruneizer.pruner.pruner.fixed_8bit = False
                
            #     return weight, weight_8, weight_4, weight_2, weight_q, bias, gate_2, gate_4, gate_8
                
                
                
        # if self._caching and not self.training and self.cached_params is None:
        #     self.cached_params = (
        #         torch.Tensor(to_numpy(weight)).to(weight.device),
        #         torch.Tensor(to_numpy(bias)).to(bias.device) if bias is not None else None,
        #     )
        return weight, weight_8, weight_4, weight_2, weight_q, bias, gate_2, gate_4, gate_8

    def get_params_saspg(self):
        # if not self.training and self.cached_params:
        #     return self.cached_params
        weight_list, gate_dict = [], {}
        weight, bias = self.get_weight_bias()
        # print('weight shape:',weight.shape)
        # import pdb;pdb.set_trace()
        # if self.training:
        # weight_and_x = [weight, x]
        # if self._prune_w:
        if self.channel_pruning:
            weight_and_bias = [weight, bias]
            weight = self.weight_pruneizer_saspg(weight_and_bias)
        else:
            weight = self.weight_pruneizer_saspg(weight)
            # weight, weight_unmask = self.weight_pruneizer_saspg(weight_and_x)\
        if isinstance(weight, list):
            bias = weight[1]
            weight = weight[0]
        
        return weight, bias
        # return weight, weight_unmask, bias

    
    def get_params_channel_vec(self):
        # if not self.training and self.cached_params:
        #     return self.cached_params
        # self.weight_pruneizer_saspg.pruner.pruner.exact_size_channel_gate = 0

        weight, bias = self.get_weight_bias()
        # print('weight shape:',weight.shape)
        # import pdb;pdb.set_trace()
        # if self.training:
        # weight_and_x = [weight, x]
        
        weight_and_bias = [weight, bias]
        weight = self.weight_pruneizer_saspg(weight_and_bias)
            
        # if self.channel_pruning:
        #     weight_and_bias = [weight, bias]
        #     weight = self.weight_pruneizer_saspg(weight_and_bias)
        # else:
        #     weight = self.weight_pruneizer_saspg(masked_weight)
        # weight, weight_unmask = self.weight_pruneizer_saspg(weight_and_x)

        """if 'q_proj' in self.layer_name or 'k_proj' in self.layer_name or 'v_proj' in self.layer_name or 'intermediate_dense' in self.layer_name:
            proj = self.channel_vector.unsqueeze(0)
            # import pdb;pdb.set_trace()
            self.weight_channel = proj * torch.abs(weight)
            self.channel_score = torch.sum(self.weight_channel, dim=1)
        elif 'out_proj' in self.layer_name or 'output_dense' in self.layer_name:
            proj = self.channel_vector.unsqueeze(1)
            self.weight_channel = proj * torch.abs(weight)
            self.channel_score = torch.sum(self.weight_channel, dim=0)
        else:
            assert self.layer_name is None
        
        # print('self.channel_score',self.channel_score)
        tau = self.weight_pruneizer_saspg.pruner.pruner.tau
        # threshold_prune_channel = self.weight_pruneizer_saspg.pruner.pruner.threshold_prune_channel
        # import pdb;pdb.set_trace()
        self.mask_channel = torch.sigmoid((self.channel_score**2 - self.threshold_prune_channel**2) / tau)
        # print('tau',tau)
        # print('threshold_prune_channel',threshold_prune_channel)
        
        # self.weight_pruneizer_saspg.pruner.pruner.threshold_prune_channel.retain_grad()
        
        
        self.mask_channel = torch.round(self.mask_channel) - self.mask_channel.detach() + self.mask_channel
        
        # print("weight torch.abs(weight) proj self.weight_channel self.channel_score, self.mask_channel threshold_prune_channel.grad:", self.threshold_prune_channel.grad, self.threshold_prune_channel)
        
        print('prune_model.channel_vector',self.channel_vector)

        if 'q_proj' in self.layer_name or 'k_proj' in self.layer_name or 'v_proj' in self.layer_name or 'intermediate_dense' in self.layer_name:
            
            masked_bias = self.mask_channel * masked_bias
            masked_channel = self.mask_channel.unsqueeze(1)
            self.weight_pruneizer_saspg.pruner.pruner.exact_size_channel_gate = torch.sum(masked_channel) * masked_weight.shape[1]
            masked_weight = masked_channel * weight
        elif 'out_proj' in self.layer_name or 'output_dense' in self.layer_name:
            masked_channel = self.mask_channel.unsqueeze(0)
            masked_weight = masked_channel * weight
            self.weight_pruneizer_saspg.pruner.pruner.exact_size_channel_gate = torch.sum(masked_channel)* masked_weight.shape[0]
            # import pdb;pdb.set_trace()
        else:
            assert self.layer_name is None"""
        
        
        if isinstance(weight, list):
            bias = weight[1]
            weight = weight[0]
        # import pdb;pdb.set_trace()
        return weight, bias
        # return self.weight_channel, bias
        # return weight, weight_unmask, bias
        
    
    # def get_params_w2(self, x):
    #     # if not self.training and self.cached_params:
    #     #     return self.cached_params
    #     weight_list, gate_dict = [], {}
    #     weight, bias = self.get_weight_bias()
    #     # print('weight:',weight)
    #     # import pdb;pdb.set_trace()
    #     # if self.training:
    #     weight_and_x = [weight, x]
    #     weight = self.weight_pruneizer_saspg(weight_and_x)
                                
    #     return weight, bias
    
    def get_params(self):
        # if not self.training and self.cached_params:
        #     return self.cached_params
        weight_list, gate_dict = [], {}
        weight, bias = self.get_weight_bias()
        if self._prune_w:
            if self.training:
                weight_list = self.weight_pruneizer(weight)
                gate_dict = self.weight_pruneizer.pruner.pruner.return_gates()
                if self.reg_type == 'disweightout' or self.reg_type == 'disweight':
                    self.weight_pruneizer.pruner.pruner.fixed_bit_dict['q8'] = 1
                    weight_list = self.weight_pruneizer(weight)
                    self.weight_pruneizer.pruner.pruner.fixed_bit_dict['q8'] = 0
                    
                    
            else:
                weight_q = self.weight_pruneizer(weight)
                
                if isinstance(weight_q, tuple):
                    weight_q = weight_q[0]
                    
                gate_dict = self.weight_pruneizer.pruner.pruner.return_gates()

                return weight, bias, weight_q, gate_dict
                
                
        # if self._caching and not self.training and self.cached_params is None:
        #     self.cached_params = (
        #         torch.Tensor(to_numpy(weight)).to(weight.device),
        #         torch.Tensor(to_numpy(bias)).to(bias.device) if bias is not None else None,
        #     )
        # import pdb;pdb.set_trace()
        return weight, bias, weight_list, gate_dict
    
    def get_weight_bias(self):
        # self.weight.requires_grad_(True)
        bias = None
        if hasattr(self, "bias"):
            bias = self.bias
        # import pdb;pdb.set_trace()
        # print('self.weight.grad:', self.weight.grad)
        return self.weight, bias

    def run_forward(self, x, weight, bias, offsets=None):
        # Performs the actual (e.g., linear) operation of the layer
        raise NotImplementedError()

    def prune_activations(self, activations, gate_dict, clip=False):
        """Pruneize a single activation tensor or all activations from a layer. I'm assuming that
        we should prune all outputs for a layer with the same pruning scheme.
        """
        if self.activation_function is not None:
            activations = self.activation_function(activations)

        if self.activation_save_target is not None:
            self.activation_save_target[self.activation_save_name] = activations.data.cpu().numpy()
        

        return activations


# class PruningHijackerWeight(PrunedModule):
#     """Mixin class that 'hijacks' the forward pass in a module to perform pruning and
#     depruning on the weights and output distributions.

#     Usage:
#     To make a pruned nn.Linear layer:
#     ```
#     >>> class PruneLinear(PruningHijacker, nn.Linear):
#     ...     pass
#     ```

#     It is vital that QSchemeForwardHijacker is the first parent class, and that the second parent
#     class derives from nn.Module, otherwise it will not be reached by a super(., .) call.

#     NB: this implementation (for now) assumes that there will always be some training involved,
#     e.g. to estimate the activation ranges.
#     """
#     def __init__(self, weight:torch.Tensor = None, bias = None, hidden_states=None, activation: nn.Module = None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if activation:
#             assert isinstance(activation, tuple(activations_list))
                    
#         self.activation_function = copy.deepcopy(activation) if activation else None

#         self.weight = weight
#         self.bias = bias
#         self.x = hidden_states
        
#         weight_prune_params = dict(n_bits=self.n_bits, scale_domain=self.scale_domain)
#         act_prune_params = dict(n_bits=self.n_bits_act, scale_domain=self.scale_domain)
            
#         if self.weight_range_method == RangeEstimators.current_minmax: 
#             # if current_minmax then weight_range_options cotaining golden section/grid will not bed used.
#             weight_init_params = dict(percentile=self.percentile)
#         else:
#             weight_init_params = self.weight_range_options
    
        
#         self.weight_pruneizer_saspg = PruningManager(
#             num_acts=self.num_acts,
#             qmethod=self.method,
#             init=self.weight_range_method,
#             per_channel=self.per_channel_weights,
#             prune_params=weight_prune_params,
#             init_params=weight_init_params,
#             gating_method=self.gating_method,
#             gate_init_dict=self.gate_init_dict,
#             checkpointing=self.checkpointing,
#             include_pruning=self.include_pruning,
#             channel_pruning=self.channel_pruning,
#             value_0_5 = self.value_0_5,
#             value_0_75 = self.value_0_75,
#             value_0_25 = self.value_0_25,
#             value_0_125 = self.value_0_125,
#             value_0_075 = self.value_0_075,
#             value_0_1 = self.value_0_1,
#             value_1 = self.value_1,
#             reg_type=self.reg_type,
#             prune_only=self.prune_only,
#             fixed_bit_dict=self.fixed_bit_dict,
#             return_bit_dict=self._return_dict,
#             fix_prob=self.fix_prob,
#             is_out_proj=self.is_out_proj,
#         )
        
#         self.activation_save_target = None
#         self.activation_save_name = None
#         self.param_list = []

#     def forward(self):
            
#         weight, bias = self.get_params_saspg()

#         return F.linear(self.x.contiguous(), weight.contiguous(), bias=bias)
        

#     def get_params_saspg(self):
#         # import pdb;pdb.set_trace()
#         print('self._prune_w',self._prune_w)
#         if self._prune_w:
#             if self.channel_pruning:
#                 weight_and_bias = [self.weight, self.bias]
#                 weight = self.weight_pruneizer_saspg(weight_and_bias)
#             else:
#                 weight = self.weight_pruneizer_saspg(self.weight)
#             if isinstance(weight, list):
#                 bias = weight[1]
#                 weight = weight[0]
        
#             return weight, bias
#         else:
#             return self.weight, self.bias
        

#     def run_forward(self, x, weight, bias, offsets=None):
#         # Performs the actual (e.g., linear) operation of the layer
#         raise NotImplementedError()

#     def prune_activations(self, activations, gate_dict, clip=False):
#         """Pruneize a single activation tensor or all activations from a layer. I'm assuming that
#         we should prune all outputs for a layer with the same pruning scheme.
#         """
#         if self.activation_function is not None:
#             activations = self.activation_function(activations)

#         if self.activation_save_target is not None:
#             self.activation_save_target[self.activation_save_name] = activations.data.cpu().numpy()
        

#         return activations
