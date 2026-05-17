# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from torch import nn

from pruning.base_pruned_classes import (
    # PrunedActivation,
    PrunedModule,
    _set_layer_learn_ranges,
    _set_layer_fix_ranges,
    _set_layer_estimate_ranges,
    _set_layer_estimate_ranges_train,
    PruningManager,
)
from pruning.autoprune_utils import PruneLinear
from transformers.modeling_utils import PreTrainedModel


class PrunedFromPretrainModel(PreTrainedModel):
    """
    Parent class for a pruned model. This allows you to have convenience functions to put the
    whole model into pruning or full precision or to freeze BN. Otherwise it does not add any
    further functionality, so it is not a necessity that a pruned model uses this class.
    """
    def get_mixed_thre_fix(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_thre_fix())

        self.apply(_fn)
        return precision_list
    
    def get_mixed_sparsity(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_sparsity())

        self.apply(_fn)
        return precision_list
    
    def get_mixed_sparsity_fix(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_sparsity_fix())

        self.apply(_fn)
        return precision_list
    
    def get_exact_size_prune(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_exact_size_prune()

        self.apply(_fn)
        return regular
    
    def get_gate_loss(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss()
        self.apply(_fn)
        return regular

    def get_gate_loss_direct(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_direct()

        self.apply(_fn)
        return regular
    
    def fix_threshold(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.fix_threshold()

        self.apply(_fn)
    
    
    def get_exact_size(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_exact_size()
                # print(regular)

        self.apply(_fn)
        return regular
        
    
    def get_mixed_prec(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_prec())

        self.apply(_fn)
        return precision_list
    
    def get_mixed_prec_fix(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_prec_fix())

        self.apply(_fn)
        return precision_list
    

    def get_gate_loss_cos(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_cos()

        self.apply(_fn)
        return regular
            
    def set_total_steps(self, total_steps):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_total_steps(total_steps)

        self.apply(_fn)
        
    def set_fix_prob(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_fix_prob()

        self.apply(_fn)
        
    def set_mag_prune(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_mag_prune()

        self.apply(_fn)
        
    def set_hand_ratio(self, hand_ratio):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_hand_ratio(hand_ratio)

        self.apply(_fn)
        
                
    def set_gumbel_hard(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_gumbel_hard()

        self.apply(_fn)
    
    def set_prob_prune(self, prob_list):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_prob_prune(prob_list)

        self.apply(_fn)
        
    def set_only_size_hard(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_only_size_hard()

        self.apply(_fn)
        
    def set_decay_tau(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_decay_tau()

        self.apply(_fn)
    
    def set_is_arc_prune(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_is_arc_prune()

        self.apply(_fn)
        
    def set_prune_2_4(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_prune_2_4()

        self.apply(_fn)
    
    def set_prune_4_8(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_prune_4_8()

        self.apply(_fn)

    def set_precision_level_direct(self,precision):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_precision_level_direct(precision)

        self.apply(_fn)

    def set_precision_level_trunc(self,precision):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_precision_level_trunc(precision)

        self.apply(_fn)
        
    def set_precision_level_mask(self,precision):
        def _fn(layer):
            if isinstance(layer, PruneLinear):
                layer.set_precision_level_mask(precision)

        self.apply(_fn)

    def pruned_weights(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.pruned_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.full_precision_weights()

        self.apply(_fn)
        

    def pruned_acts(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.pruned_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def pruned(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.pruned()

        self.apply(_fn)

    def full_precision(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.full_precision()

        self.apply(_fn)
    
    
    def return_unmask(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_unmask()

        self.apply(_fn)
        
    def return_mask(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_mask()

        self.apply(_fn)
        
    def return_fp(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_fp()

        self.apply(_fn)

    def return_q(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q()

        self.apply(_fn)
    
    def return_q8(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q8()

        self.apply(_fn)

    def return_q4(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q4()

        self.apply(_fn)
        
    def return_q5(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q5()

        self.apply(_fn)
        
    def return_q2(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q2()

        self.apply(_fn)

    def return_q3(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q3()

        self.apply(_fn)

    # Methods for switching pruner pruning states
    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def fix_act_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'activation_pruner'):
                _set_layer_fix_ranges(module.activation_pruner)

        self.apply(_fn)

    def fix_weight_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'weight_pruneizer'):
                _set_layer_fix_ranges(module.weight_pruneizer)

        self.apply(_fn)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_act_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'activation_pruner'):
                _set_layer_estimate_ranges(module.activation_pruner)

        self.apply(_fn)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def reset_act_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'activation_pruner'):
                module.activation_pruner.reset_ranges()

        self.apply(_fn)

    def set_prune_state(self, weight_prune, act_prune):
        # assert not weight_prune and not act_prune
        if act_prune:
            self.pruned_acts()
        else:
            self.full_precision_acts()

        if weight_prune:
            self.pruned_weights()
        else:
            self.full_precision_weights()

    def get_gate_loss_prune(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_prune_channel(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune_channel()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_prune_channel10(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune_channel10()
                # print(regular)

        self.apply(_fn)
        return regular
    
class PrunedModel(nn.Module):
    """
    Parent class for a pruned model. This allows you to have convenience functions to put the
    whole model into pruning or full precision or to freeze BN. Otherwise it does not add any
    further functionality, so it is not a necessity that a pruned model uses this class.
    """
    def get_gate_loss(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss()
                # print(regular)
        self.apply(_fn)
        return regular

    def get_gate_loss_direct(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_direct()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_prune(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_prune_channel_gate(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune_channel_gate()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_prune_channel(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            # import pdb;pdb.set_trace()
            # print(type(layer))
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune_channel()
                # print('regular', regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_prune_channel10(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_prune_channel10()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def fix_threshold(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.fix_threshold()

        self.apply(_fn)
    
    def get_gate_loss_distil(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_distil()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_disweight(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_disweight()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_disLinear(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_disLinear()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_disweightout(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_disweightout()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_exact_size(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_exact_size()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_exact_size_prune(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_exact_size_prune()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_mse_weight(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_mse_weight()
                # print(regular)

        self.apply(_fn)
        return regular
        
    
    def get_mixed_prec(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_prec())
                # print(regular)

        self.apply(_fn)
        return precision_list
    
    def get_mixed_prec_fix(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_prec_fix())
                # print(regular)

        self.apply(_fn)
        return precision_list
    
    def get_mixed_sparsity(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_sparsity())
                # print(regular)

        self.apply(_fn)
        return precision_list
    
    def get_mixed_sparsity_fix(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_sparsity_fix())
                # print(regular)

        self.apply(_fn)
        return precision_list
    
    def get_mixed_thre_fix(self):
        precision_list = []
        def _fn(layer):
            nonlocal precision_list
            if isinstance(layer, PruningManager):
                precision_list.append(layer.get_mixed_thre_fix())
                # print(regular)

        self.apply(_fn)
        return precision_list
    
    
    def get_gate_loss_KL(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_KL()
                # print(regular)

        self.apply(_fn)
        return regular
    
    def get_gate_loss_KL_out(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_KL_out()
                # print(regular)

        self.apply(_fn)
        return regular

    def get_gate_loss_L1(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_L1()
                # print(regular)

        self.apply(_fn)
        return regular

    def get_gate_loss_cos(self):
        regular = 0
        def _fn(layer):
            nonlocal regular
            if isinstance(layer, PruningManager):
                regular += layer.get_gate_loss_cos()
                # print(regular)

        self.apply(_fn)
        return regular
    # def set_mac(self, mac):
    #     def _fn(layer):
    #         if isinstance(layer, PruningManager):
    #             layer.set_mac(mac)

    #     self.apply(_fn)
    
    # def set_delta(self, weight_dict):
    #     def _fn(layer):
    #         if isinstance(layer, PruningManager):
    #             layer.set_delta(weight_dict)

    #     self.apply(_fn)
            
    def set_total_steps(self, total_steps):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_total_steps(total_steps)

        self.apply(_fn)
        
    def set_fix_prob(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_fix_prob()

        self.apply(_fn)
        
    def set_mag_prune(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_mag_prune()

        self.apply(_fn)
        
    def set_hand_ratio(self, hand_ratio):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_hand_ratio(hand_ratio)

        self.apply(_fn)
                  
    def set_gumbel_hard(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_gumbel_hard()

        self.apply(_fn)
        
    def set_prob_prune(self, prob_list):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_prob_prune(prob_list)

        self.apply(_fn)
        
    def set_only_size_hard(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_only_size_hard()

        self.apply(_fn)

    def set_decay_tau(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_decay_tau()

        self.apply(_fn)
        
    def set_prune_2_4(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_prune_2_4()

        self.apply(_fn)
    
    def set_prune_4_8(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_prune_4_8()

        self.apply(_fn)
    
    def set_is_arc_prune(self):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_is_arc_prune()

        self.apply(_fn)
        
    def set_precision_level_direct(self,precision):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_precision_level_direct(precision)

        self.apply(_fn)

    def set_precision_level_trunc(self,precision):
        def _fn(layer):
            if isinstance(layer, PruningManager):
                layer.set_precision_level_trunc(precision)

        self.apply(_fn)
        
    def set_precision_level_mask(self,precision):
        def _fn(layer):
            if isinstance(layer, PruneLinear):
                layer.set_precision_level_mask(precision)

        self.apply(_fn)
    # def linear_set_precision_level(self,precision):
    #     def _fn(layer):
    #         if isinstance(layer, PruneLinear):
    #             layer.linear_set_precision_level(precision)

    #     self.apply(_fn)

    def pruned_weights(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.pruned_weights()

        self.apply(_fn)

    def full_precision_weights(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.full_precision_weights()

        self.apply(_fn)
        

    def pruned_acts(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.pruned_acts()

        self.apply(_fn)

    def full_precision_acts(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.full_precision_acts()

        self.apply(_fn)

    def pruned(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.pruned()

        self.apply(_fn)

    def full_precision(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.full_precision()

        self.apply(_fn)

    def return_unmask(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_unmask()

        self.apply(_fn)
        
    def return_mask(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_mask()

        self.apply(_fn)
        
    def return_fp(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_fp()

        self.apply(_fn)

    def return_q(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q()

        self.apply(_fn)
    
    def return_q8(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q8()

        self.apply(_fn)

    def return_q4(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q4()

        self.apply(_fn)
        
    def return_q2(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q2()

        self.apply(_fn)

    def return_q3(self):
        def _fn(layer):
            if isinstance(layer, PrunedModule):
                layer.return_q3()

        self.apply(_fn)

    # Methods for switching pruner pruning states
    def learn_ranges(self):
        self.apply(_set_layer_learn_ranges)

    def fix_ranges(self):
        self.apply(_set_layer_fix_ranges)

    def fix_act_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'activation_pruner'):
                _set_layer_fix_ranges(module.activation_pruner)

        self.apply(_fn)

    def fix_weight_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'weight_pruneizer'):
                _set_layer_fix_ranges(module.weight_pruneizer)

        self.apply(_fn)

    def estimate_ranges(self):
        self.apply(_set_layer_estimate_ranges)

    def estimate_act_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'activation_pruner'):
                _set_layer_estimate_ranges(module.activation_pruner)

        self.apply(_fn)

    def estimate_ranges_train(self):
        self.apply(_set_layer_estimate_ranges_train)

    def reset_act_ranges(self):
        def _fn(module):
            if isinstance(module, PrunedModule) and hasattr(module, 'activation_pruner'):
                module.activation_pruner.reset_ranges()

        self.apply(_fn)

    def set_prune_state(self, weight_prune, act_prune):
        # assert not weight_prune and not act_prune
        if act_prune:
            self.pruned_acts()
        else:
            self.full_precision_acts()

        if weight_prune:
            self.pruned_weights()
        else:
            self.full_precision_weights()
