# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from collections import namedtuple
from enum import Enum

import torch
from torch.autograd import Function
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
import copy
from torch import autograd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ThresholdBinarizer(Function):
    """
    Threshold Binarizer with learnable threshold.
    Computes a binary mask M such that `M_{i,j} = 1` if `S_{i,j}^2 > threshold^2`, else 0.
    Uses Straight-Through Estimator (STE) and gradient approximation via sigmoid.

    Args:
        inputs (torch.Tensor): Input matrix (e.g., weight matrix)
        threshold (torch.Tensor): Learnable threshold (scalar)
        tau (float): Temperature parameter for gradient approximation
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, threshold: torch.Tensor, tau: float = 1.0):
        # 保存中间结果用于反向传播
        ctx.save_for_backward(inputs, threshold)
        ctx.tau = tau

        # 计算平方差和概率掩码
        squared_diff = inputs.pow(2) - threshold.pow(2)
        prob_mask = torch.sigmoid(squared_diff / tau)  # 用于梯度近似
        
        mask = (prob_mask > 0.5).type(inputs.type())
        # 保存概率掩码用于反向传播
        ctx.save_for_backward(inputs, threshold, prob_mask)
        return prob_mask
    
    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的中间结果
        inputs, threshold, prob_mask = ctx.saved_tensors
        tau = ctx.tau

        # 计算概率掩码对平方差的梯度
        grad_squared_diff = grad_output * prob_mask * (1 - prob_mask) / tau

        # 计算 inputs 的梯度: dL/dinputs = grad_squared_diff * 2 * inputs
        grad_inputs = grad_squared_diff * 2 * inputs

        # 计算 threshold 的梯度: dL/dthreshold = grad_squared_diff * (-2) * threshold
        # grad_threshold = grad_squared_diff * (-2) * threshold

        return grad_inputs, None, None  # tau 不需要梯度
    
class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer.
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

# class ThresholdBinarizer(autograd.Function):
#     """
#     Thresholdd binarizer.
#     Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > \tau`
#     where `\tau` is a real value threshold.

#     Implementation is inspired from:
#         https://github.com/arunmallya/piggyback
#         Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
#         Arun Mallya, Dillon Davis, Svetlana Lazebnik
#     """

#     @staticmethod
#     def forward(ctx, inputs: torch.tensor, threshold: float, tau: bool):
#         """
#         Args:
#             inputs (`torch.FloatTensor`)
#                 The input matrix from which the binarizer computes the binary mask.
#             threshold (`float`)
#                 The threshold value (in R).
#             sigmoid (`bool`)
#                 If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
#                 In this case, `threshold` should be a value between 0 and 1.
#         Returns:
#             mask (`torch.FloatTensor`)
#                 Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
#                 retained, 0 - the associated weight is pruned).
#         """
        
#         mask = (torch.sigmoid((inputs**2 - threshold**2) / tau)).type(inputs.type())
        
#         return mask

#     @staticmethod
#     def backward(ctx, gradOutput):
#         return gradOutput, None, None
      
class SparseGreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, torch.tensor(threshold))
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        input, threshold, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<threshold] = 0
        return grad_input, None

class GreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return torch.Tensor.float(torch.gt(input, threshold))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        The backward behavior of the floor function is defined as the identity function.
        """
        grad_input = grad_output.clone()
        return grad_input, None



class RoundStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


class FloorStraightThrough(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad


round_ste_func = RoundStraightThrough.apply
floor_ste_func = FloorStraightThrough.apply


class PrunerBase(nn.Module):
    def __init__(self, n_bits, per_channel=False, axis=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.axis = axis

    @property
    def is_initialized(self):
        raise NotImplementedError()

    @property
    def x_max(self):
        raise NotImplementedError()

    @property
    def symmetric(self):
        raise NotImplementedError()

    @property
    def x_min(self):
        raise NotImplementedError()

    def forward(self, x_float):
        raise NotImplementedError()

    def _adjust_params_per_axis(self, x):
        raise NotImplementedError()

    def _adjust_params_per_channel(self, x):
        raise NotImplementedError()

    def set_prune_range(self, x_min, x_max):
        raise NotImplementedError()

    def get_full_class_name(self):
        return f"{self.__module__}.{self.__class__.__name__}"
    
    def extra_repr(self):
        return (
            f'n_bits={self.n_bits}, per_channel={self.per_channel}, axis={self.axis}, '
            f'is_initialized={self.is_initialized}' # 子类调用父类的属性 有delta就是true
        )

    def reset(self):
        self._delta = None

def gatesigmoid(x, thre, temp):
    return torch.sigmoid((x**2- thre**2) / temp)

def hardsigmoid(x, beta=2.0 / 3.0, zeta=1.1, gamma=-0.1, N=None):
    sigm = torch.sigmoid(x / beta)
    ybar = sigm * (zeta - gamma) + gamma
    return torch.clamp(ybar, 0, 1)

def invsigmoid(x, beta=2.0 / 3.0):
    return -beta * np.log(1.0 / x - 1)

def l0_sample(log_alpha, beta=2.0 / 3.0, zeta=1.1, gamma=-0.1, N=1):
    u = torch.rand(N if N > 1 else log_alpha.size(0)).cuda()
    sigm = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    sbar = sigm * (zeta - gamma) + gamma
    
    return torch.clamp(sbar, 0, 1)


def hc_prob_pos(p, beta=2.0 / 3.0, zeta=1.1, gamma=-0.1):
    return torch.sigmoid(p - beta * math.log(-gamma / zeta))

def get_x_min_x_max(x, use_mse=False):
    x_min, x_max = x.min().item(), x.max().item()
    if not use_mse or True:
        return x_min, x_max


class ParentPruner(nn.Module):
    """ A general pruner class which gets a pruning function and then takes care of
    the statistics etc. Depending on the parameters it keeps a running mean or takes the current
    matrix to get the necessary statistics to initialize the pruning function.

    Parameters
    ----------
    method: str
        Name of the pruning method to use.
    n_bits:
        Number of bits for pruning.
    use_running_mean:
        If true it keeps a running mean of the matrix statistics, otherwise it uses alwats the
        statistics of the current matrix or pruning.
    momentum:
        The momentum for the running mean.

    """

    def __init__(
        self,
        method,
        n_bits=8,
        num_acts=1,
        fix_range_on_first_forward=True,
        scale_domain="linear",
        gating_method="pg",
        gate_init_dict=None,
        act_prune=False,
        checkpointing=False,
        include_pruning=False,
        channel_pruning=False,
        value_0_75 = 0.,
        value_0_5 = 0.,
        value_0_25 = 0.,
        value_0_125 = 0.,
        value_1 = 0.,
        value_0_1 = 0.,
        value_0_075 = 0.,
        prune_only=False,
        fixed_bit_dict=None,
        reg_type="const",
        gate_dict=None,
        return_bit_dict=None,
        fix_prob=None,
        is_out_proj=False,
    ):
        
        super(ParentPruner, self).__init__()

        self.method = method
        self.scale_domain = scale_domain
        
        self.n_bits = n_bits
        self.num_acts = num_acts
        self.act_prune = act_prune
        
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
                
        self.prune_only = prune_only
        assert not prune_only or include_pruning, "{}; {}; {}".format(
            prune_only, include_pruning, act_prune
        )
        self.fixed_bit_dict = fixed_bit_dict
        self.reg_type = reg_type

        self.is_out_proj = is_out_proj
        self.x_min = None
        self.x_max = None
        self.pruner = None
        self.fixed_range = False
        self.fix_range_on_first_forward = fix_range_on_first_forward

        self.gating_method = gating_method
        
        self.gate_init_dict = gate_init_dict
        self.fixed_bit_dict = fixed_bit_dict
        
        self.gate_dict = gate_dict
        self.return_bit_dict = return_bit_dict
        
        self.fix_prob = fix_prob
        
        self._on = True
        self.owner = None
        self.name = None
        
        if "bayesian_bits" in self.method or "pact_only" in self.method:
            if self.pruner is None:
                self.pruner = self.create_pruner()
                
    @property
    def symmetric(self):
        return not self.act_prune
       
    @property
    def is_initialized(self):
        raise NotImplementedError()


    def off(self):
        self._on = False
        self.pruner.do_reg = False

    def on(self):
        self._on = True
        self.pruner.do_reg = True

    def create_pruner(self):
        if self.method == "bayesian_bits":
           
            assert not self.prune_only , "{}; {}; {}".format(
                self.prune_only, self.act_prune
            )
            
            if self.act_prune:
                # activation
                q = AsyBayesianBitsPruner(
                    num_acts=self.num_acts,
                    n_bits=self.n_bits,
                    gating_method=self.gating_method,
                    gate_init_dict=self.gate_init_dict,
                    act_prune=self.act_prune,
                    checkpointing=self.checkpointing,
                    include_pruning=self.include_pruning,
                    channel_pruning=self.channel_pruning,
                    value_0_5 = self.value_0_5,
                    value_1 = self.value_1,
                    value_0_75 = self.value_0_75,
                    value_0_25 = self.value_0_25,
                    value_0_125 = self.value_0_125,
                    value_0_1 = self.value_0_1,
                    value_0_075 = self.value_0_075,
                    prune_only=self.prune_only,
                    fixed_bit_dict=self.fixed_bit_dict,
                    reg_type=self.reg_type,
                    gate_dict=self.gate_dict,
                    return_bit_dict=self.return_bit_dict,
                    fix_prob=self.fix_prob,
                    is_out_proj=self.is_out_proj,
                )
            else:
                q = SyBayesianBitsPruner(
                    num_acts=self.num_acts,
                    n_bits=self.n_bits,
                    gating_method=self.gating_method,
                    gate_init_dict=self.gate_init_dict,
                    act_prune=self.act_prune,
                    checkpointing=self.checkpointing,
                    include_pruning=self.include_pruning,
                    channel_pruning=self.channel_pruning,
                    value_0_5 = self.value_0_5,
                    value_0_75 = self.value_0_75,
                    value_0_25 = self.value_0_25,
                    value_0_125 = self.value_0_125,
                    value_1 = self.value_1,
                    value_0_1 = self.value_0_1,
                    value_0_075 = self.value_0_075,
                    prune_only=self.prune_only,
                    fixed_bit_dict=self.fixed_bit_dict,
                    reg_type=self.reg_type,
                    return_bit_dict=self.return_bit_dict,
                    fix_prob=self.fix_prob,
                    is_out_proj=self.is_out_proj,
                )
            
        else:
            raise ValueError("Unknown method: {}".format(self.method))
        
        return q

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name

    def forward(self, x_float):
        if not self._on:
            return x_float
            
        return self.pruner(x_float)

    def extra_repr(self):
        s = "{method}, n_bits={n_bits}"
        return s.format(**self.__dict__)
    

class PerChannelPruneizeWrapper(nn.Module):
    """ A wrapper that does a per channel pruning (one set of parameters per channel).

    Parameters
    ----------
    pruner: Pruner
        An initialized pruner that will be used per channel. This will be cloned for each channel
        so it keeps track of their own statistics.

    """

    def __init__(self, pruner):
        super().__init__()
        self.org_pruner = pruner
        self.channel_pruners = None

    def forward(self, x_float):
        if self.channel_pruners is None:
            self.channel_pruners = [
                copy.deepcopy(self.org_pruner) for _ in range(x_float.shape[0])
            ]

        x_prune = torch.zeros_like(x_float)
        for idx in range(x_float.shape[0]):
            x_prune[idx] = self.channel_pruners[idx](x_float[idx])
        return x_prune
    

class AsymmetricUniformPruner(PrunerBase):
    """
    PyTorch Module that implements Asymmetric Uniform Pruning using STE.
    Pruneizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the pruning that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for pruning.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel pruning
    """
    def __init__(self, n_bits, scale_domain='linear', per_channel=False, axis=None, eps=1e-8, extra_bits=8, clip=False):

        super().__init__(n_bits, per_channel)

        assert scale_domain in ('linear', 'log')
        delta_list = ['_delta','_delta_5bit', '_delta_4bit', '_delta_2bit']

        zero_float_list = ['_zero_float','_zero_float_5bit','_zero_float_4bit', '_zero_float_2bit']

        for bit in delta_list:
            self.register_buffer(bit, None)
        for bit in zero_float_list:
            self.register_buffer(bit, None)


        
        self.n_bits = n_bits
        self.scale_domain = scale_domain
        self.per_channel = per_channel
        self.axis = axis
        self.eps = eps
        self.extra_bits = extra_bits
        self.clip_val = None

    

    
    # A few useful properties

    @property
    def delta(self):
        if self._delta is not None:
            return self._delta # tensor(0.1310, device='cuda:0')
 
        else:
            raise PrunerNotInitializedError()

    @property
    def delta_4bit(self):
        if self._delta_4bit is not None:
            return self._delta_4bit # tensor(0.1310, device='cuda:0')
        
        else:
            raise PrunerNotInitializedError()
        
    @property
    def delta_5bit(self):
        if self._delta_5bit is not None:
            return self._delta_5bit # tensor(0.1310, device='cuda:0')
 
        else:
            raise PrunerNotInitializedError()
        
    @property
    def delta_2bit(self):
        if self._delta_2bit is not None:
            return self._delta_2bit # tensor(0.1310, device='cuda:0')
 
        else:
            raise PrunerNotInitializedError()

    @property
    def delta_3bit(self):
        if self._delta_3bit is not None:
            return self._delta_3bit # tensor(0.1310, device='cuda:0')
 
        else:
            raise PrunerNotInitializedError()
    
        
    @property
    def zero_float(self):
        if self._zero_float is not None:
            return self._zero_float # tensor(136.2763, device='cuda:0')qq
        else:
            raise PrunerNotInitializedError()
    
    @property
    def zero_float_2bit(self):
        if self._zero_float_2bit is not None:
            return self._zero_float_2bit # tensor(136.2763, device='cuda:0')qq
        else:
            raise PrunerNotInitializedError()

    @property
    def zero_float_4bit(self):
        if self._zero_float_4bit is not None:
            return self._zero_float_4bit # tensor(136.2763, device='cuda:0')qq
        else:
            raise PrunerNotInitializedError()
    
    @property
    def zero_float_6bit(self):
        if self._zero_float_6bit is not None:
            return self._zero_float_6bit # tensor(136.2763, device='cuda:0')qq
        else:
            raise PrunerNotInitializedError()
        
    @property
    def zero_float_5bit(self):
        if self._zero_float_5bit is not None:
            return self._zero_float_5bit # tensor(136.2763, device='cuda:0')qq
        else:
            raise PrunerNotInitializedError()
        
    @property
    def is_initialized(self):
        # only if all the deltas is not None
        # 0610 change to if any of the deltas is True then true
        if hasattr(self, '_delta'):
            if self._delta:
                return True
            
        if hasattr(self, '_delta_4bit'):
            if self._delta_4bit:
                return True

        if hasattr(self, '_delta_2bit'):
            if self._delta_2bit:
                return True
        
        if hasattr(self, '_delta_3bit'):
            if self._delta_3bit:
                return True

        return False
        
    @property
    def symmetric(self):
        return False

    
    def int_min(self, n_bits=None):
        # integer grid minimum
        return 0.0

    
    def int_max(self, n_bits=None):
        # integer grid maximum
        if n_bits is None:
            return 2.0 ** self.n_bits - 1
        else:
            return 2.0 ** n_bits - 1

    @property
    def scale(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta)

    @property
    def scale_2bit(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta_2bit, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta_2bit)

    @property
    def scale_3bit(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta_3bit, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta_3bit)
        
    @property
    def scale_4bit(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta_4bit, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta_4bit)
        
    @property
    def scale_6bit(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta_6bit, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta_6bit)
        
    @property
    def scale_5bit(self):
        if self.scale_domain == 'linear':
            return torch.clamp(self.delta_5bit, min=self.eps)
        elif self.scale_domain == 'log':
            return torch.exp(self.delta_5bit)
        
    @property
    def zero_point(self):
        zero_point = round_ste_func(self.zero_float)
        zero_point = torch.clamp(zero_point, self.int_min(8), self.int_max(8))
        return zero_point

    @property
    def zero_point_3bit(self):
        zero_point_3bit = round_ste_func(self.zero_float_3bit)
        zero_point_3bit = torch.clamp(zero_point_3bit, self.int_min(3), self.int_max(3))
        return zero_point_3bit

    @property
    def zero_point_2bit(self):
        zero_point_2bit = round_ste_func(self.zero_float_2bit)
        zero_point_2bit = torch.clamp(zero_point_2bit, self.int_min(2), self.int_max(2))
        return zero_point_2bit
    
    @property
    def zero_point_4bit(self):
        zero_point_4bit = round_ste_func(self.zero_float_4bit)
        zero_point_4bit = torch.clamp(zero_point_4bit, self.int_min(4), self.int_max(4))
        return zero_point_4bit
    
    @property
    def zero_point_5bit(self):
        zero_point_5bit = round_ste_func(self.zero_float_5bit)
        zero_point_5bit = torch.clamp(zero_point_5bit, self.int_min(6), self.int_max(6))
        return zero_point_5bit
    
    @property
    def zero_point_6bit(self):
        zero_point_6bit = round_ste_func(self.zero_float_6bit)
        zero_point_6bit = torch.clamp(zero_point_6bit, self.int_min(6), self.int_max(6))
        return zero_point_6bit
    
    def to_integer_forward(self, x_float, fixed_scale=None, n_bits=None):
        """
        Qunatized input to its integer represantion
        Parameters
        ----------
        x_float: PyTorch Float Tensor
                Full-precision Tensor

        Returns
        -------
        x_int: PyTorch Float Tensor of integers
        """
        if n_bits not in [2, 3, 4, 5, 6, 8]: # 8-bit -> zero_point
            raise NotImplementedError()
        zero_point = getattr(self, f"zero_point_{n_bits}bit", self.zero_point)
        x_int = round_ste_func(x_float / fixed_scale) + zero_point
        x_int = torch.clamp(x_int, self.int_min(n_bits), self.int_max(n_bits))

        return x_int

    
    def forward(self, x_float):
        """
        Pruneizes (pruned to integer and the scales back to original domain)
        Parameters
        ----------
        x_float: PyTorch Float Tensor
            Full-precision Tensor

        Returns
        -------
        x_prune: PyTorch Float Tensor
            Pruned-Depruned Tensor
        """
        if self.axis is not None:
            self._adjust_params_per_axis(x_float)

        if self.per_channel:
            self._adjust_params_per_channel(x_float)

        assert self.extra_bits == 8
        
        if  self.extra_bits == 8:
            
            if self.n_bits == 8: # or self.n_bits == 4 or self.n_bits == 6:
                x_int = self.to_integer_forward(x_float, fixed_scale=self.scale, n_bits = self.n_bits)
                x_prune = self.scale * (x_int - self.zero_point)
               
            elif self.n_bits == 4:
                x_int = self.to_integer_forward(x_float,  fixed_scale=self.scale_4bit, n_bits = self.n_bits)
                x_prune = self.scale_4bit * (x_int - self.zero_point_4bit)
            
            elif self.n_bits == 3:
                x_int = self.to_integer_forward(x_float,  fixed_scale=self.scale_3bit, n_bits = self.n_bits)
                x_prune = self.scale_3bit * (x_int - self.zero_point_3bit)

            elif self.n_bits == 6:
                x_int = self.to_integer_forward(x_float,  fixed_scale=self.scale_6bit, n_bits = self.n_bits)
                x_prune = self.scale_6bit * (x_int - self.zero_point_6bit)
                
            elif self.n_bits == 5:
                x_int = self.to_integer_forward(x_float,  fixed_scale=self.scale_5bit, n_bits = self.n_bits)
                x_prune = self.scale_5bit * (x_int - self.zero_point_5bit)
                
            elif self.n_bits == 2:
                x_int = self.to_integer_forward(x_float, fixed_scale=self.scale_2bit, n_bits = self.n_bits)
                x_prune = self.scale_2bit * (x_int - self.zero_point_2bit)
        
        return x_prune
    

    def _adjust_params_per_axis(self, x_float):
        r = len(x_float.size())
        new_shape = [1] * self.axis + [-1] + [1] * (r - self.axis - 1)
        self._delta = self._delta.view(new_shape)
        self._zero_float = self._zero_float.view(new_shape)

    def _adjust_params_per_channel(self, x):
        """
        Adjusts the pruning parameter tensors (delta, zero_float)
        to the input tensor shape if they don't match

        Parameters
        ----------
        x: input tensor
        """
        if x.ndim != self.delta.ndim:
            new_shape = [-1] + [1] * (len(x.shape) - 1)
            self._delta = self.delta.view(new_shape)
            if self._zero_float is not None:
                self._zero_float = self._zero_float.view(new_shape)

    def _tensorize_min_max(self, x_min, x_max):
        """
        Converts provided min max range into tensors
        Parameters
        ----------
        x_min: float or PyTorch 1D tensor
        x_max: float or PyTorch 1D tensor

        Returns
        -------
        x_min: PyTorch Tensor 0 or 1-D
        x_max: PyTorch Tensor 0 or 1-D
        """
        # Ensure a torch tensor
        if not torch.is_tensor(x_min):
            x_min = torch.tensor(x_min).float()
            x_max = torch.tensor(x_max).float()

        if x_min.dim() > 0 and len(x_min) > 1 and not self.per_channel and self.axis is None:
            raise ValueError(
                'x_min and x_max must be a float or 1-D Tensor'
                ' for per-tensor pruning (per_channel=False)'
            )
        # Ensure we always use zero and avoid division by zero
        x_min = torch.min(x_min, torch.zeros_like(x_min))
        x_max = torch.max(x_max, torch.ones_like(x_max) * self.eps)
        x_min = x_min.cuda()
        x_max = x_max.cuda()
        
        return x_min, x_max

    def set_prune_range(self, x_min, x_max, signal_bits):
        """
        Instantiates the pruning parameters based on the provided
        min and max range

        Parameters
        ----------
        x_min: tensor or float
                Pruning range minimum limit
        x_max: tensor of float
                Pruning range minimum limit
        """
        x_min, x_max = self._tensorize_min_max(x_min, x_max)
        
        # Define the mapping from signal_bits to attribute names
        attribute_names = {
            8: ['_delta', '_zero_float'],
            # 5: ['_delta_5bit','_zero_float_5bit'],
            # 6: ['_delta_6bit','_zero_float_6bit'],
            4: ['_delta_4bit','_zero_float_4bit'],
            # 3: ['_delta_3bit','_zero_float_3bit'],
            2: ['_delta_2bit','_zero_float_2bit'],
        }

        # Get the attribute name for the given signal_bits
        attr_delta = attribute_names.get(signal_bits[0])
        attr_zero_float = attribute_names.get(signal_bits[1])
        
        if attr_delta is None:
            raise NotImplementedError()

        # Perform the operations
        delta = getattr(self, attr_delta.lstrip('_'))
        _delta = (x_max - x_min) / self.int_max(signal_bits)
        _zero_float = (-x_min / delta).detach()
        
        if self.scale_domain == 'log':
            _delta = torch.log(delta)
            
        delta = delta.detach()

        # Set the attribute
        setattr(self, attr_delta, _delta)
        setattr(self, attr_zero_float, _zero_float)

    def make_range_trainable(self):
    # Define the parameters
        parameters = {
            '_delta': [self.delta, '_zero_float'],
            # '_delta_5bit': [self.delta_5bit, '_zero_float_5bit'],
            '_delta_4bit': [self.delta_4bit, '_zero_float_4bit'],
            # '_delta_3bit': [self.delta_3bit, '_zero_float_3bit'],
            '_delta_2bit': [self.delta_2bit, '_zero_float_2bit'],
        }

        # Iterate over the parameters
        for param_name, param_value in parameters.items():
            _param_delta = getattr(self, param_name)
            _param_zero = getattr(self, param_value[1])
            # Check if the parameter is not already a nn.Parameter
            if not any(torch.equal(param, param_value[0]) for param in self.parameters()):
                # Convert the parameter to a nn.Parameter
                setattr(self, param_name, torch.nn.Parameter(_param_delta))
                setattr(self, param_value[1], torch.nn.Parameter(_param_zero))
                

class SymmetricUniformPruner(AsymmetricUniformPruner):
    """
    PyTorch Module that implements Symmetric Uniform Pruning using STE.
    Pruneizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the pruning that occurred.

    Parameters
    ----------
    n_bits: int
        Number of bits for pruning.
    scale_domain: str ('log', 'linear) with default='linear'
        Domain of scale factor
    per_channel: bool
        If True: allows for per-channel pruning
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('_signed', None)
    @property
    def signed(self):
        if self._signed is not None:
            return self._signed.item()
        else:
            raise PrunerNotInitializedError()
    
    
    @property
    def symmetric(self):
        return True


    def int_min(self,n_bits=None):
        if n_bits is None:
            return -(2.0 ** (self.n_bits - 1)) if self.signed else 0
        else:
            return -(2.0 ** (n_bits - 1)) if self.signed else 0


    def int_max(self, n_bits=None):
        if n_bits is None:
            pos_n_bits = self.n_bits - self.signed
            return 2.0 ** pos_n_bits - 1
        else:
            pos_n_bits = n_bits - self.signed
            return 2.0 ** pos_n_bits - 1

    @property
    def zero_point(self):
        return 0.0

    @property
    def zero_point_4bit(self):
        return 0.0

    @property
    def zero_point_3bit(self):
        return 0.0
    
    @property
    def zero_point_6bit(self):
        return 0.0
    
    @property
    def zero_point_5bit(self):
        return 0.0
    
    @property
    def zero_point_2bit(self):
        return 0.0
    
    def set_prune_range(self, x_min, x_max, signal_bits):
        """
        Instantiates the pruning parameters based on the provided
        min and max range

        Parameters
        ----------
        x_min: tensor or float
                Pruning range minimum limit
        x_max: tensor of float
                Pruning range minimum limit
        """
        x_min, x_max = self._tensorize_min_max(x_min, x_max)

        self._signed = x_min.min() < 0

        x_absmax = torch.max(x_min.abs(), x_max)
        
        # Define the mapping from signal_bits to attribute names
        attribute_names = {
            8: '_delta',
            5: '_delta_5bit',
            # 6: '_delta_6bit',
            4: '_delta_4bit',
            # 3: '_delta_3bit',
            2: '_delta_2bit',
        }

        # Get the attribute name for the given signal_bits
        attr_name = attribute_names.get(signal_bits)
        
        if attr_name is None:
            raise NotImplementedError()

        # Perform the operations
        _delta = x_absmax / self.int_max(signal_bits)
        if self.scale_domain == 'log':
            _delta = torch.log(_delta)
        _delta = _delta.detach()

        # Set the attribute
        setattr(self, attr_name, _delta)
        
    
    def make_range_trainable(self):
    # Define the parameters
        if self.n_bits == 8:
            parameters = {
                '_delta': self.delta,
                
            }
        elif self.n_bits == 4:
            parameters = {
                '_delta_4bit': self.delta_4bit,
               
            }
        elif self.n_bits == 2:
            parameters = {
                '_delta_2bit': self.delta_2bit,
            }

        # Iterate over the parameters
        for param_name, param_value in parameters.items():
            _param_value = getattr(self, param_name)
            # Check if the parameter is not already a nn.Parameter
            if not any(torch.equal(param, param_value) for param in self.parameters()):
                # Convert the parameter to a nn.Parameter
                setattr(self, param_name, torch.nn.Parameter(_param_value))
                

    
class AsyBayesianBitsPruner(AsymmetricUniformPruner):
    def __init__(
        self,
        n_bits,
        num_acts=1,
        cuda=True,
        gating_method="pg",
        gate_init_dict=None,
        act_prune=False,
        checkpointing=False,
        include_pruning=False,
        channel_pruning=False,
        value_0_5 = 0.,
        value_0_25 = 0.,
        value_0_75 = 0.,
        value_0_125 = 0.,
        value_1 = 0.,
        value_0_1 = 0.,
        value_0_075 = 0.,
        prune_only=False,
        fixed_bit_dict=None,
        reg_type="const",
        gate_dict=None,
        return_bit_dict=None,
        fix_prob=None,
        is_out_proj=False,
        *args, **kwargs
        # per_channel=False,
    ):
        # learned_scale can be None, 'scale' or 'range'
        super().__init__(n_bits, *args, **kwargs)

        self.mixed_precision = 0
        self.act_prune = act_prune
        self.checkpointing = checkpointing
        
        self.include_pruning = include_pruning
        self.channel_pruning = channel_pruning
        
        self.value_0_5 = value_0_5
        self.value_0_25 = value_0_25
        self.value_1 = value_1
        self.value_0_75 = value_0_75
        self.value_0_125 = value_0_125
        self.value_0_1 = value_0_1
        self.value_0_075 = value_0_075
        
        
        self.prune_only = prune_only
       
        assert not self.prune_only, "{}; {}; {}".format(
            self.prune_only, self.act_prune
        )
        self.gate_init_dict = gate_init_dict
        self.fixed_bit_dict = fixed_bit_dict
       
        self.mac_count, self.max_macs = None, None
        self.do_reg = True
        self.reg_type = reg_type
 

        self.is_out_proj = is_out_proj

        self.cuda = cuda
        
        self.threshold_2 = None
        self.threshold_4 = None
        self.threshold_3 = None
        self.threshold_8 = None
        
        self.gt = GreaterThan.apply
                
        self.gating_method = gating_method
        
        self.calib = 0
        self.num_acts = num_acts
        self.len_mixed_precision = 3 # len of mixed_precision
        
        self.gate_dict = gate_dict
        
        self.use_L1 = False
        self.use_cos = False
        
        self.use_distil = False
        self.distil_loss = 0.

    def regularizer_size(self): # part of the final loss

        return 0.
    
    def regularizer_size_prune(self): # part of the final loss

        return 0.

    def regularizer_KL(self):
        reg_KL = 0.

        return reg_KL

    def get_exact_size(self):
        return 0.
    
    def get_exact_size_prune(self):
        return 0.
        
    def regularizer_KL_out(self):
        reg_KL = 0.

        return reg_KL

    def regularizer_distil(self):
        reg_cos = 0.
       
        return reg_cos

    def regularizer_disweight(self):
        reg_L1 = 0.
        return reg_L1
    
    def regularizer_disweightout(self):
        reg_L1 = 0.
        return reg_L1
    
    def regularizer_mse_pruned(self):
        reg_L1 = 0.
        return reg_L1
    
    def regularizer_disLinear(self):
        reg_L1 = 0.
        return reg_L1

    def get_gates(self,x_fp2, x_fp4, x_fp8, x_fp16, x_fp32, N=1):
        pass

    
    def forward(self, x):
        pass

class SyBayesianBitsPruner(SymmetricUniformPruner):
    def __init__(
        self,
        n_bits,
        num_acts=1,
        cuda=True,
        gating_method="pg",
        gate_init_dict=None,
        act_prune=False,
        checkpointing=False,
        include_pruning=False,
        channel_pruning=False,
        value_0_5 = 0.,
        value_1 = 0.,
        value_0_75 = 0.,
        value_0_25 = 0.,
        value_0_125 = 0.,
        value_0_1 = 0.,
        value_0_075 = 0.,
        prune_only=False,
        fixed_bit_dict=None,
        reg_type="const",
        return_bit_dict=None,
        fix_prob=None,
        is_out_proj=False,
        *args, **kwargs
        # per_channel=False,
    ):
        # learned_scale can be None, 'scale' or 'range'
        super().__init__(n_bits, *args, **kwargs)

        self.n_bits = n_bits
        # self.per_channel = per_channel
        self.mixed_precision = 0
        self.act_prune = act_prune
        self.checkpointing = checkpointing
        
        self.include_pruning = include_pruning
        
        self.channel_pruning = channel_pruning
        # self.prune_output = True
        
        
        self.gate_init_dict = gate_init_dict
        self.fixed_bit_dict = fixed_bit_dict

        self.mac_count, self.max_macs = None, None
        self.do_reg = True
        self.reg_type = reg_type

        self.return_bit_dict = return_bit_dict
        
        self.cuda = cuda
        
        if 'q' in self.gate_init_dict:         
            self.thre_init_dict = float(self.gate_init_dict['q'])
        else:
            self.thre_init_dict = { key:float(value) for key, value in self.gate_init_dict.items()}
        
        parameters = {
            'threshold': {'init_value': self.thre_init_dict, 'indices': [2, 4, 8]},
        }

        for param, info in parameters.items():
            for index in info['indices']:

                if f'q{index}' in info['init_value'] and isinstance(index, int):
                    tensor = torch.tensor([info['init_value'][f'q{index}']]).to("cuda" if self.cuda else "cpu")
                    setattr(self, f'{param}_{index}', tensor)
                    setattr(self, f'{param}_{index}', torch.nn.Parameter(tensor))
                    
                elif isinstance(index, str) and 'q' + '_' + index in info['init_value']:
                    ffn_index = 'q' + '_' + index
                    tensor = torch.tensor([info['init_value'][ffn_index]]).to("cuda" if self.cuda else "cpu")
                    setattr(self, f'{param}_{index}', tensor)
                    setattr(self, f'{param}_{index}', torch.nn.Parameter(tensor))
                    
                else:
                    import pdb;pdb.set_trace()
                    assert index in ['ffn_2', 2]


        self.KL_8_4 = None
        self.KL_8_2 = None
        self.KL_4_2 = None
        
        self.KL_q_fp = None
        
        self.mixed_bit = None
        self.sp_ratio = None
        
        self.use_KL_8_4 = True
        self.use_KL_8_2 = True
        
        self.gt = GreaterThan.apply
                
        self.gating_method = gating_method
                
        self.calib = 0
        self.num_acts = num_acts
        self.len_mixed_precision = 3

        self.x_shape = None
        self.x_shape_0 = None
        self.x_shape_1 = None
        
        self.use_L1 = True
        self.use_cos = False
        
        self.use_distil = False
        self.distil_loss = 0.
        
        self.disweightout_loss = 0.
        
        self.gate_dict = None

        
        self.is_fixed_threshold_2 = 1
        self.is_fixed_threshold_4 = 0
        self.is_fixed_threshold_8 = 0
        
        self.is_fixed_threshold_ffn_2 = 1
        self.is_fixed_threshold_ffn_4 = 0
        self.is_fixed_threshold_ffn_8 = 0
        
        self.current_epoch = 0
        self.current_steps = 0
        
        self.total_steps = 0
        
        # self.t0 = 1.
        self.t0 = 0.5 # 2024.11.2 14:57
        self.eta_max = self.t0
        self.n0 = 0.
        
        self.phi = 1e-5
        
        # self.eta_min  = 0.03 # 2024.7.5 22:35 从0.05改成0.03
        self.eta_min = 0.01 # 2024.11.2 14:57 0.5->0.01
         
        self.fix_prob = False # will apply set_fix_prob() in main_prune
        
        self.mag_prune = False
        self.hand_ratio = False
        
        
        self.is_hard = False
        self.only_size_hard = False
        
        self.decay_tau = False
        self.tau = None
        
        self.is_arc_prune = False
        
        self.exp_prob = None
        self.argmax_index = None
        self.x_prune_mask_init = False
        self.x_prune_mask_hard = None

        self.prune_2_4 = False
        self.prune_4_8 = False
       
        self.bias_shape = None
        
        
        self.is_out_proj = is_out_proj
        
        self.mask = None
        self.mag_mask = None
        
        tensor = torch.tensor(1e-5).to("cuda" if self.cuda else "cpu")
        # tensor_a_b = torch.tensor(1.).to("cuda" if self.cuda else "cpu")
        
        # setattr(self, f'a', torch.nn.Parameter(tensor_a_b))
        # setattr(self, f'b', torch.nn.Parameter(tensor_a_b))

        setattr(self, f'threshold_prune', torch.nn.Parameter(tensor))

        # tensor = torch.tensor(1e-6).to("cuda" if self.cuda else "cpu")
        # setattr(self, f'threshold_prune_channel', torch.nn.Parameter(tensor))
        
        self.layer_name = None
        
        # self.threshold_prune_channel = torch.nn.Parameter(torch.tensor(30.),requires_grad=True).to('cuda')
        self.mask_channel = None
        self.channel_score = None
        # self.channel_score = torch.tensor(1.).to('cuda')
        
        self.register_buffer('current_ratio', torch.tensor(1.))
        self.register_buffer('prune_output', torch.tensor(0.))
        
        self.mask_1 = None
        self.mask_0_75 = None
        self.mask_0_5 = None
        self.mask_0_25 = None
        self.mask_0_125 = None
        self.mask_0_1 = None
        self.mask_0_075 = None
                
    def print_module_names(self):
        for name, module in self.named_modules():
            print(f'Module Name: {name}')

    
    def get_exact_size(self):
        if self.x_shape != 768*3072 and self.x_shape != 768*768 and self.x_shape != 1024*1024 and self.x_shape != 4096*1024:
            raise NotImplementedError
        return self.exact_size
    
    def get_exact_size_prune(self):
        # print(self.exact_size)
        return self.exact_size

    def output_mixed_prec(self):
        if self.x_shape == 768*768 or self.x_shape == 768*3072 or self.x_shape == 1024*1024 or self.x_shape == 4096*1024:
            if self.x_shape_1 is not None:
                self.mixed_bit.append([self.x_shape_0, self.x_shape_1])
            else:
                self.mixed_bit.append([self.x_shape_0])

        return self.mixed_bit
    
    def output_mixed_prec_fix(self):
        if self.x_shape == 768*3072 or self.x_shape == 768*768 or self.x_shape == 1024*1024 or self.x_shape == 4096*1024:
            if self.x_shape_1 is not None:
                return {self.mixed_bit:[self.x_shape_0, self.x_shape_1]}
            else:
                return {self.mixed_bit:[self.x_shape_0]}
            
    
    def output_mixed_sparsity_fix(self):
        if self.x_shape == 768*3072 or self.x_shape == 768*768 or self.x_shape == 1024*1024 or self.x_shape == 4096*1024:
            if self.x_shape_1 is not None:
                return {self.sp_ratio:[self.x_shape_0, self.x_shape_1]}
            else:
                return {self.sp_ratio:[self.x_shape_0]}
            
    def output_mixed_thre_fix(self):
        if self.x_shape == 768*3072 or self.x_shape == 768*768 or self.x_shape == 1024*1024 or self.x_shape == 4096*1024:
            if self.x_shape_1 is not None:
                return {float(self.threshold_prune* 1000):[self.x_shape_0, self.x_shape_1]}
            else:
                return {float(self.threshold_prune* 1000):[self.x_shape_0]}
        
    def regularizer_size(self): # part of the final loss
        
        reg_size = 0.
        
        size = self.x_shape / 8

        if self.exp_prob is not None:
            argmax_index = torch.argmax(self.exp_prob)
            self.exact_size = 2**(argmax_index + 1) * self.x_shape # 8*size
            
            if self.fix_prob:
                reg_size = (self.exact_size)
            else:
                # 2024.9.24 modified always not use only_size_hard here; before it is an option
                reg_size = (self.gumble_prob[0] * 2 * size + self.gumble_prob[1] * 4 * size + self.gumble_prob[2] * 8 * size) # first pruning then prune, exact size after pruning is hard/not-hard before sending to prune?

        return reg_size
    
    def regularizer_size_prune_channel(self): # part of the final loss
        # import pdb;pdb.set_trace()
        reg_size = 0.
        
        if self.prune_output:
            size = self.x_shape + self.bias_shape
        else:
            size = self.x_shape

        # SASPG str: no Gumbel ladder; exact_size set from active channel mask in the layer forward.
        if self.value_0_075 == 0 and self.value_0_1 == 0:
            return float(getattr(self, "exact_size", 0) or 0)
        
        # num_ones = torch.sum(self.x_prune_mask_hard == 1).item()
        # num_zeros = torch.sum(self.x_prune_mask_hard == 0).item()
        # assert num_ones * 3 == num_zeros, "The ratio of the number of 1s to 0s is not 1:3"

        # import pdb;pdb.set_trace()
        if self.exp_prob is not None:
            
            # index_to_ratio = {0:1, 1:0.5, 2:0.25, 3:0.125, 4:0.75}
            if self.x_shape == 768*768:
                index_to_ratio = {0:1, 1:0.5, 2:0.25, 3:1/12, 4:0.75}
            elif self.x_shape == 4096*1024:
                index_to_ratio = {0:1, 1:0.5, 2:0.25, 3:0.125, 4:0.75, 5:409/4096, 6:307/4096}
            elif self.x_shape == 3072*768:
                index_to_ratio = {0:1, 1:0.5, 2:0.25, 3:0.125, 4:0.75, 5:307/3072, 6:230/3072}
            elif self.x_shape == 1024*1024:
                index_to_ratio = {0:1, 1:0.5, 2:0.25, 3:0.125, 4:0.75, 5:1/16}
            else:
                raise NotImplementedError()

            self.argmax_index = int(torch.argmax(self.exp_prob))
            
            self.sp_ratio = index_to_ratio[int(self.argmax_index)]
            # assert int(self.argmax_index) == 0
            
            if self.prune_output:
                self.exact_size = self.sp_ratio*(self.x_shape + self.bias_shape)
               
            else:
                self.exact_size = self.sp_ratio*(self.x_shape)
                
            if self.fix_prob:
                reg_size = (self.exact_size)
            else:
                # 2024.9.24 modified always not use only_size_hard here; before it is an option
                if self.x_shape == 768*768:
                    reg_size = (self.gumble_prob[0] * 1 * size + self.gumble_prob[1] * 0.5 * size + self.gumble_prob[2] * 0.25 * size + self.gumble_prob[3] * 1/12 * size + self.gumble_prob[4] * 0.75 * size)
                elif self.x_shape == 768*3072:
                    reg_size = (
                        self.gumble_prob[0] * 1 * size
                        + self.gumble_prob[1] * 0.5 * size
                        + self.gumble_prob[2] * 0.25 * size
                        + self.gumble_prob[3] * 0.125 * size
                        + self.gumble_prob[4] * 0.75 * size
                        + self.gumble_prob[5] * 307/3072 * size
                        + self.gumble_prob[6] * 230/3072 * size
                    )
                elif self.x_shape == 1024 * 1024:
                    reg_size = (
                        self.gumble_prob[0] * 1 * size
                        + self.gumble_prob[1] * 0.5 * size
                        + self.gumble_prob[2] * 0.25 * size
                        + self.gumble_prob[3] * 0.125 * size
                        + self.gumble_prob[4] * 0.75 * size
                        + self.gumble_prob[5] * 1/16 * size
                    )
                elif self.x_shape == 4096 * 1024:
                    reg_size = (
                        self.gumble_prob[0] * 1 * size
                        + self.gumble_prob[1] * 0.5 * size
                        + self.gumble_prob[2] * 0.25 * size
                        + self.gumble_prob[3] * 0.125 * size
                        + self.gumble_prob[4] * 0.75 * size
                        + self.gumble_prob[5] * 409/4096 * size
                        + self.gumble_prob[6] * 307/4096 * size
                    )
                else:
                    raise NotImplementedError()
                #     or self.x_shape == 1024*1024:
                #     if self.value_0_75 == 0.:
                #         reg_size = (self.gumble_prob[0] * 1 * size + self.gumble_prob[1] * 0.5 * size + self.gumble_prob[2] * 0.25 * size + self.gumble_prob[3] * 1/12 * size) 
                #     else:
                #         reg_size = (self.gumble_prob[0] * 1 * size + self.gumble_prob[1] * 0.5 * size + self.gumble_prob[2] * 0.25 * size + self.gumble_prob[3] * 1/12 * size + self.gumble_prob[4] * 0.75 * size) # first pruning then prune, exact size after pruning is hard/not-hard before sending to prune?
                # elif self.x_shape == 4096*1024 or self.x_shape == 3072*768:
                #     if self.value_0_75 == 0.:
                #         reg_size = (self.gumble_prob[0] * 1 * size + self.gumble_prob[1] * 0.5 * size + self.gumble_prob[2] * 0.25 * size + self.gumble_prob[3] * 0.125 * size) 
                #     else:
                #         reg_size = (self.gumble_prob[0] * 1 * size + self.gumble_prob[1] * 0.5 * size + self.gumble_prob[2] * 0.25 * size + self.gumble_prob[3] * 0.125 * size + self.gumble_prob[4] * 0.75 * size) # first pruning then prune, exact size after pruning is hard/not-hard before sending to prune?
                               
        return reg_size
    
    def regularizer_size_prune_channel10(self): # part of the final loss
        
        reg_size = 0.
        
        if self.prune_output:
            size = self.x_shape + self.bias_shape
        else:
            size = self.x_shape

        # num_ones = torch.sum(self.x_prune_mask_hard == 1).item()
        # num_zeros = torch.sum(self.x_prune_mask_hard == 0).item()
        # assert num_ones * 3 == num_zeros, "The ratio of the number of 1s to 0s is not 1:3"

        if self.exp_prob is not None:
            
            index_to_ratio = {0:1, 1:0.5, 2:0.25, 3:0.125, 4:0.75, 5:0.1, 6:0.075} # 2024/12/14 21:00 change to 6 elements
            
            self.sp_ratio = index_to_ratio[int(self.argmax_index)]
            
            if self.prune_output:
                self.exact_size = self.sp_ratio*(self.x_shape + self.bias_shape)
               
            else:
                self.exact_size = self.sp_ratio*(self.x_shape)
            
            if self.fix_prob:
                reg_size = (self.exact_size)
            else:
                # first pruning then prune, exact size after pruning is hard/not-hard before sending to prune?
                reg_size = (self.gumble_prob[0] * 1 * size + self.gumble_prob[1] * 0.5 * size + self.gumble_prob[2] * 0.25 * size + self.gumble_prob[3] * 0.125 * size + self.gumble_prob[4] * 0.75 * size + self.gumble_prob[5] * 0.1 * size + self.gumble_prob[6] * 0.075 * size) # first pruning then prune, exact size after pruning is hard/not-hard before sending to prune?            
               
        return reg_size
    
    def regularizer_size_prune_channel_gate(self):
        # print('self.exact_size_channel_gate:',self.exact_size_channel_gate, self.exact_size_channel_gate//768)
        return self.exact_size_channel_gate
    
    
    def regularizer_size_prune(self): # part of the final loss
        # self.exact_size = 1
        return self.exact_size

    # def regularizer_size_prune_channel_prune(self):
    #     return self.exact_size
    
    def return_gates(self):
        return self.gate_dict
    
    # def forward_channel_pruning_trying_but_not_used(self, x):
        
    #     self.exact_size_channel_gate = 0


    #     if isinstance(x, list):
    #         bias = x[1]
    #         x = x[0]
            
    #     if self.x_shape is None:
    #         self.x_shape = x.numel()
    #         self.x_shape_0 = x.size(0)
    #         if  len(x.size()) == 2:
    #             self.x_shape = x.size(0) * x.size(1)
    #             self.x_shape_1 = x.size(1)
    #             self.x_shape_list = [x.size(0),x.size(1)] 
    #         elif len(x.size()) == 1:
    #             self.x_shape = x.size(0)
    #             self.x_shape_list = [x.size(0)]

    #         elif len(x.size()) == 3:
    #             self.x_shape_1 = x.size(1)
    #             self.x_shape_2 = x.size(2)
    #             self.x_shape = x.size(0) * x.size(1) * x.size(2)
    #             self.x_shape_list = [x.size(0),x.size(1),x.size(2)] 
    #         else:
    #             raise NotImplementedError()

          
        
    #     """if self.calib < self.num_acts * self.len_mixed_precision and not self.training:

    #         x_prune = super(SyBayesianBitsPruner, self).forward(x)
    #         self.calib += 1
    #         print('calibing...')
    #         return [x_prune]"""
        
    #     if self.training:
    #         assert self.decay_tau
    #     if self.decay_tau:
    #         if self.total_steps > 0:
    #             # import pdb;pdb.set_trace()
    #             print('self.eta_min',self.eta_min)
    #             print('self.eta_max',self.eta_max)
    #             self.tau = self.eta_min + 0.5 * (self.t0 - self.eta_min) * (1 + math.cos(math.pi * self.current_steps / self.total_steps)) # 0.5 --> 0.01
    #         else:
    #             self.tau = 0.25
    #     else:
    #         self.tau = 0.25
        
    #     # import pdb;pdb.set_trace()
    #     if self.training:
    #         self.current_steps += 1
        
        
    #     # self.mask = torch.sigmoid((x**2 - self.threshold_prune**2) / self.tau)
        
    #     """
    #     if 'q_proj' in self.layer_name or 'k_proj' in self.layer_name or 'v_proj' in self.layer_name or 'intermediate_dense' in self.layer_name:
    #         # proj = self.channel_vector.unsqueeze(0)
    #         # import pdb;pdb.set_trace()
    #         # self.weight_channel = proj * torch.abs(x)
    #         # self.weight_channel = proj * (x**2)
    #         # self.weight_channel = (x**2)
    #         # # self.weight_channel = proj * x
    #         # self.channel_score = torch.sqrt(torch.sum(self.weight_channel, dim=1))
    #         if self.value_1 == 100:  
    #             # print('abs in')  
    #             self.channel_score = (torch.abs(x).sum(dim=1))**2
    #         # square_x = x**2
    #         elif self.value_1 == 50:
    #             W_squared = x ** 2          # 计算每个元素的平方

    #             # 提取目标统计量
    #             # target_min = W_squared.min().item()
    #             # target_max = W_squared.max().item()
    #             target_min = 0
    #             target_max = 1
    #             target_var = torch.var(W_squared, unbiased=False).item()  # 有偏方差

    #             # 计算L2范数向量（假设对每行求范数）
    #             v =  ((x**2).sum(dim=1))  # shape=[128]

    #             # 应用方案一
    #             self.channel_score = self.scale_range_then_variance(v, target_min, target_max, target_var)
    #         else:
    #             self.channel_score = ((x**2).sum(dim=1))
    #             # print('w2 in')  
    #         # self.channel_score = self.channel_score / self.channel_score.sum()
            
    #     elif 'out_proj' in self.layer_name or 'output_dense' in self.layer_name:
    #         # proj = self.channel_vector.unsqueeze(1)
    #         # self.weight_channel = proj * torch.abs(x)
    #         # self.weight_channel = proj * (x**2)
    #         # self.weight_channel = (x**2)
    #         # self.channel_score = torch.sqrt(torch.sum(self.weight_channel, dim=0))
    #         if self.value_1 == 100:
    #             # print('abs out')  
    #             self.channel_score = (torch.abs(x).sum(dim=0))**2
    #         elif self.value_1 == 50:
    #             W_squared = x ** 2          # 计算每个元素的平方

    #             # 提取目标统计量
    #             # target_min = W_squared.min().item()
    #             # target_max = W_squared.max().item()
    #             target_min = 0
    #             target_max = 1
    #             target_var = torch.var(W_squared, unbiased=False).item()  # 有偏方差

    #             # 计算L2范数向量（假设对每行求范数）
    #             v =  ((x**2).sum(dim=0))  # shape=[128]

    #             # 应用方案一
    #             self.channel_score = self.scale_range_then_variance(v, target_min, target_max, target_var)
    #         else:
    #         # square_x = x**2
    #             # print('w2')  
    #             self.channel_score = ((x**2).sum(dim=0))
    #         # self.channel_score = self.channel_score / self.channel_score.sum()
    #     else:
    #         assert self.layer_name is None
        
    #     # print('self.channel_score',self.channel_score)
    #     # threshold_prune_channel = self.weight_pruneizer_saspg.pruner.pruner.threshold_prune_channel
    #     # import pdb;pdb.set_trace()
    #     # self.mask_channel = torch.sigmoid((self.channel_score**2 - self.threshold_prune_channel**2) / (self.tau))
    #     # # print('tau',tau)
    #     # # print('threshold_prune_channel',threshold_prune_channel)
        
    #     # # self.weight_pruneizer_saspg.pruner.pruner.threshold_prune_channel.retain_grad()
        
        
    #     # self.mask_channel = torch.round(self.mask_channel) - self.mask_channel.detach() + self.mask_channel
        
    #     # print("weight torch.abs(weight) proj self.weight_channel self.channel_score, self.mask_channel threshold_prune_channel.grad:", self.threshold_prune_channel.grad, self.threshold_prune_channel)
    #     # print('self.threshold_prune_channel',self.threshold_prune_channel)
    #     # print('self.channel_vector',self.channel_vector)

    #     if 'q_proj' in self.layer_name or 'k_proj' in self.layer_name or 'v_proj' in self.layer_name or 'intermediate_dense' in self.layer_name:
            
    #         # masked_bias = self.mask_channel * bias
    #         # masked_channel = self.mask_channel.unsqueeze(1)
    #         # x = masked_channel * x
            
    #         if self.mask_channel is None:
    #             masked_bias = bias
    #             self.exact_size_channel_gate = self.x_shape
    #         else:
    #             # import pdb;pdb.set_trace()
    #             masked_bias = self.mask_channel * bias
    #             masked_channel_for_weight = self.mask_channel.unsqueeze(1)
    #             x = masked_channel_for_weight * x
    #             self.exact_size_channel_gate = torch.sum(self.mask_channel) * float(x.shape[1])
    #             # print('sp in', torch.sum(self.mask_channel)/self.mask_channel.numel(), self.mask_channel.numel())
    #     elif 'out_proj' in self.layer_name or 'output_dense' in self.layer_name:
    #         # masked_channel = self.mask_channel.unsqueeze(0)
    #         # x = masked_channel * x
    #         # print('self.layer_name', self.layer_name, self.mask_channel)
    #         if self.mask_channel is None:
    #             self.exact_size_channel_gate = self.x_shape
    #         else:
    #             # print('torch.sum(self.mask_channel)',torch.sum(self.mask_channel))
    #             masked_channel_for_weight = self.mask_channel.unsqueeze(0)
    #             x = masked_channel_for_weight * x
    #             self.exact_size_channel_gate = torch.sum(self.mask_channel) * float(x.shape[0])
    #             # print('sp out', torch.sum(self.mask_channel)/self.mask_channel.numel(), self.mask_channel.numel())
    #         # import pdb;pdb.set_trace()
    #     else:
    #         assert self.layer_name is None
            
    #     if 'q_proj' in self.layer_name or 'k_proj' in self.layer_name or 'v_proj' in self.layer_name or 'intermediate_dense' in self.layer_name:
    #         x_out = [x, masked_bias]
    #         return x_out
    #     else:
    #         return x
    #     """
    #     return x
    def create_mask(self, x, threshold_ratio=0.25):
        # 计算 x 的绝对值
        abs_x = torch.abs(x)
        
        # 展平并排序绝对值
        sorted_abs_x, _ = torch.sort(abs_x.flatten(), descending=True)
        
        # 确定 25% 的阈值位置
        k = int(threshold_ratio * len(sorted_abs_x))
        threshold_value = sorted_abs_x[k]
        
        # 创建掩码
        mask = torch.where(abs_x >= threshold_value, torch.ones_like(x), torch.zeros_like(x))
        return mask
    
    def forward(self, x):
    # def forward(self, x):
        
        if isinstance(x, list):
            bias = x[1]
            x = x[0]
            
        if self.x_shape is None:
            self.x_shape = x.numel()
            self.x_shape_0 = x.size(0)
            if  len(x.size()) == 2:
                self.x_shape = x.size(0) * x.size(1)
                self.x_shape_1 = x.size(1)
                self.x_shape_list = [x.size(0),x.size(1)] 
            elif len(x.size()) == 1:
                self.x_shape = x.size(0)
                self.x_shape_list = [x.size(0)]

            elif len(x.size()) == 3:
                self.x_shape_1 = x.size(1)
                self.x_shape_2 = x.size(2)
                self.x_shape = x.size(0) * x.size(1) * x.size(2)
                self.x_shape_list = [x.size(0),x.size(1),x.size(2)] 
            else:
                raise NotImplementedError()
            
        # import pdb;pdb.set_trace()
        # print('self.x_shape',self.x_shape)
        
        """if self.calib < self.num_acts * self.len_mixed_precision and not self.training:

            x_prune = super(SyBayesianBitsPruner, self).forward(x)
            self.calib += 1
            print('calibing...')
            return [x_prune]"""
        # import pdb;pdb.set_trace()
        # if self.training:
        #     assert self.decay_tau
        # if self.decay_tau:
        if not self.channel_pruning:
            if self.total_steps > 0:
                # print('self.eta_min',self.eta_min)
                # print('self.eta_max',self.eta_max)
                # self.tau = self.eta_min + 0.5 * (self.t0 - self.eta_min) * (1 + math.cos(math.pi * self.current_steps / self.total_steps)) # 0.5 --> 0.01
                self.tau = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.current_steps / self.total_steps)) # 0.5 --> 0.01
            else:
                self.tau = self.eta_min
        # else:
        #     self.tau = 0.25
        if self.channel_pruning:
            
            self.warmup_ratio = 0.1

            if self.training:
                
                if self.eta_max == self.eta_min:
                    self.tau = self.eta_max
                else:
                    progress = self.current_steps / self.total_steps
                    warmup = progress <= self.warmup_ratio

                    if warmup:
                        ratio = progress / self.warmup_ratio
                        cos_term = 1
                    else:
                        ratio = (progress - self.warmup_ratio) / (1 - self.warmup_ratio)
                        cos_term = -1

                    self.tau = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + cos_term * math.cos(math.pi * ratio))

                    # print('self.total_steps:',self.total_steps)
            else:
                self.tau = self.eta_max
        
        # import pdb;pdb.set_trace()
        if self.training:
            self.current_steps += 1
            
        # return x
    
        # assert not self.channel_pruning
        if self.channel_pruning:
            if self.bias_shape is None:
                self.bias_shape = bias.numel()
                # import pdb;pdb.set_trace()
                # print('self.bias_shape',self.bias_shape)

            # if self.channel_prune10:
            #     self.exp_prob = torch.cat([self.prob_1, self.prob_0_5, self.prob_0_25, self.prob_0_125, self.prob_0_75, self.prob_0_1, self.prob_0_075])
            # else:
            #     if self.value_0_75 == 0.:
            #         raise NotImplementedError()
            #     else:
            #         self.exp_prob = torch.cat([self.prob_1, self.prob_0_5, self.prob_0_25, self.prob_0_125, self.prob_0_75])
                
            # self.gumble_prob = torch.nn.functional.gumbel_softmax(self.exp_prob, tau=self.tau, hard=self.is_hard)
            # if self.training:      
            #     if self.prune_output:
            #         if self.channel_prune10:
            #             x_prune = self.gumble_prob[0] * self.mask_1 * x + self.gumble_prob[1] * self.mask_0_5 * x  + self.gumble_prob[2] * self.mask_0_25 * x + self.gumble_prob[3] * self.mask_0_125 * x + self.gumble_prob[4] * self.mask_0_75 * x + self.gumble_prob[5] * self.mask_0_1 * x + self.gumble_prob[6] * self.mask_0_075 * x
                            
            #             bias_prune = self.gumble_prob[0] * self.mask_1.squeeze() * bias + self.gumble_prob[1] * self.mask_0_5.squeeze() * bias  + self.gumble_prob[2] * self.mask_0_25.squeeze() * bias + self.gumble_prob[3] * self.mask_0_125.squeeze() * bias + self.gumble_prob[4] * self.mask_0_75.squeeze() * bias + self.gumble_prob[5] * self.mask_0_1.squeeze() * bias + self.gumble_prob[6] * self.mask_0_075.squeeze() * bias
            #         else:
            #             if self.value_0_75 == 0.:
            #                 raise NotImplementedError()
            #             else:
            #                 x_prune = self.gumble_prob[0] * self.mask_1 * x + self.gumble_prob[1] * self.mask_0_5 * x  + self.gumble_prob[2] * self.mask_0_25 * x + self.gumble_prob[3] * self.mask_0_125 * x + self.gumble_prob[4] * self.mask_0_75 * x
                            
            #                 bias_prune = self.gumble_prob[0] * self.mask_1.squeeze() * bias + self.gumble_prob[1] * self.mask_0_5.squeeze() * bias  + self.gumble_prob[2] * self.mask_0_25.squeeze() * bias + self.gumble_prob[3] * self.mask_0_125.squeeze() * bias + self.gumble_prob[4] * self.mask_0_75.squeeze() * bias
                    
            #     else:
            #         if self.channel_prune10:
            #             x_prune = self.gumble_prob[0] * self.mask_1.T * x + self.gumble_prob[1] * self.mask_0_5.T * x  + self.gumble_prob[2] * self.mask_0_25.T * x + self.gumble_prob[3] * self.mask_0_125.T * x + self.gumble_prob[4] * self.mask_0_75.T * x + self.gumble_prob[5] * self.mask_0_1.T * x + self.gumble_prob[6] * self.mask_0_075.T * x
            #         else:
            #             if self.value_0_75 == 0.:
            #                 raise NotImplementedError()
            #             else:
            #                 x_prune = self.gumble_prob[0] * self.mask_1.T * x + self.gumble_prob[1] * self.mask_0_5.T * x  + self.gumble_prob[2] * self.mask_0_25.T * x + self.gumble_prob[3] * self.mask_0_125.T * x + self.gumble_prob[4] * self.mask_0_75.T * x
            # else:
            #     self.argmax_index = int(torch.argmax(self.exp_prob))


            #     if self.prune_output:
            #         if self.argmax_index == 0:
            #             x_prune = self.mask_1 * x
            #             bias_prune = self.mask_1.squeeze() * bias
            #         elif self.argmax_index == 1:
            #             x_prune = self.mask_0_5 * x
            #             bias_prune = self.mask_0_5.squeeze() * bias
            #         elif self.argmax_index == 2:
            #             x_prune = self.mask_0_25 * x
            #             bias_prune = self.mask_0_25.squeeze() * bias
            #         elif self.argmax_index == 3:
            #             x_prune = self.mask_0_125 * x
            #             bias_prune = self.mask_0_125.squeeze() * bias
            #         elif self.argmax_index == 4:
            #             x_prune = self.mask_0_75 * x
            #             bias_prune = self.mask_0_75.squeeze() * bias
            #         elif self.argmax_index == 5:
            #             x_prune = self.mask_0_1 * x
            #             bias_prune = self.mask_0_1.squeeze() * bias
            #         elif self.argmax_index == 6:
            #             x_prune = self.mask_0_075 * x
            #             bias_prune = self.mask_0_075.squeeze() * bias
            #     else: 
            #         if self.argmax_index == 0:
            #             x_prune = self.mask_1.T * x
            #         elif self.argmax_index == 1:
            #             x_prune = self.mask_0_5.T * x
            #         elif self.argmax_index == 2:
            #             x_prune = self.mask_0_25.T * x
            #         elif self.argmax_index == 3:
            #             x_prune = self.mask_0_125.T * x
            #         elif self.argmax_index == 4:
            #             x_prune = self.mask_0_75.T * x
            #         elif self.argmax_index == 5:
            #             x_prune = self.mask_0_1.T * x
            #         elif self.argmax_index == 6:
            #             x_prune = self.mask_0_075.T * x
                    
            
            # self.argmax_index = torch.argmax(self.exp_prob)
            
            # if self.prune_output:
            #     x_out = [x_prune, bias_prune]
            #     return x_out
            # else:
            #     return x_prune
            return x
                
        if not (self.fix_prob or self.mag_prune):
            self.mask = torch.sigmoid((x**2 - self.threshold_prune**2) / self.tau)
            # self.mask = (x**2 / ( x**2 + self.tau))
        
        if self.mag_prune:
            
            if self.mask is None and self.mag_mask is None:
                
                if self.hand_ratio == 0.:
    
                    self.sp_ratio = float(self.current_ratio)
                    # load的是ratio不是threshold，没有负数
                    x_flat_abs = torch.abs(x).flatten()

                    # 对展平后的绝对值 tensor 进行排序
                    x_flat_abs_sorted, _ = torch.sort(x_flat_abs,descending=True)

                    # 计算 75% 位置的值
                    k = int(self.current_ratio * len(x_flat_abs_sorted)) - 1
                    if k <0:
                        k=0
                    threshold_value = x_flat_abs_sorted[k]

                    self.mask = torch.where(torch.round(torch.sigmoid((x**2 - threshold_value**2)/self.eta_min)) ==1. , torch.ones_like(x), torch.zeros_like(x)) # change to this 2024.12.16 12:11
                    
                else:
                    self.mask = self.create_mask(x, self.hand_ratio)
                    
                    inverted_mask = 1 - self.mask
                    x_masked_prune = inverted_mask * x
                    max_value_masked = torch.max(torch.abs(x_masked_prune.flatten()))
        
                    x_prune = self.mask * x
                    x_prune_flat = x_prune.flatten()
                    m = x_prune_flat != 0
                    
                    min_value_saved = torch.min(torch.abs(x_prune_flat[m]))
                
                    assert max_value_masked <= min_value_saved
            
            if self.mag_mask is None:
                x_prune = self.mask * x
                
                count_ones = torch.sum(torch.eq(self.mask, 1)).item()
                self.sp_ratio = count_ones/self.mask.numel()
                self.exact_size = count_ones
            
            else:
                # print('self.mag_mask used')
                x_prune = self.mag_mask * x
                
                count_ones = torch.sum(torch.eq(self.mag_mask, 1)).item()
                self.sp_ratio = count_ones/self.mag_mask.numel()
                self.exact_size = count_ones
            
            # print(self.sp_ratio)
                        
        elif self.fix_prob: # Here the functio nof fix_prob is equal to it in if not self.training below; fix_prob only for evaluation
            if self.mask is None:
                self.mask = torch.where(torch.round(torch.sigmoid((x**2 - self.threshold_prune**2)/self.tau)) ==1. , torch.ones_like(x), torch.zeros_like(x)) # >= to > 2024.11.10 01:19
            
            x_prune = self.mask * x 

            
        else:
            if not self.training:
                
                # self.binary_mask = torch.where(torch.abs(x) > torch.abs(self.threshold_prune), torch.ones_like(x), torch.zeros_like(x)) # >= to > 2024.11.10 01:19 # change to this 2024.12.16 12:11
                # self.mask = torch.round((x**2 / ( x**2 + self.tau)))
                # self.binary_mask = torch.where(torch.round(torch.sigmoid((x**2 - self.threshold_prune**2)/self.eta_min)) ==1. , torch.ones_like(x), torch.zeros_like(x)) # change to this 2024.12.16 12:11
                # print('not self.training')
                self.binary_mask = torch.where(torch.round(torch.sigmoid((x**2 - self.threshold_prune**2)/self.tau)) ==1. , torch.ones_like(x), torch.zeros_like(x))

                # import pdb;pdb.set_trace()
                """self.binary_mask = TopKBinarizer.apply(
                    self.mask_scores, self.threshold_prune
                )"""
                # self.mask = ThresholdBinarizer.apply(
                #     x, self.threshold_prune, self.tau)
                
                # self.binary_mask = torch.round(self.mask)
                self.exact_size = torch.sum(self.binary_mask) 

                x_prune = self.binary_mask * x
                

                count_ones = torch.sum(torch.eq(self.binary_mask, 1)).item()

                self.sp_ratio = count_ones/self.binary_mask.numel()

              
            else:
                # import pdb;pdb.set_trace()
                assert self.only_size_hard
                
                """self.mask = TopKBinarizer.apply(
                    self.mask_scores, self.threshold_prune
                )"""
                # print('x.shape:',x.shape)
                # self.mask = ThresholdBinarizer.apply(
                #     x, self.threshold_prune, self.tau)
                
                if self.only_size_hard:
                    self.mask = torch.round(self.mask) - self.mask.detach() + self.mask # hard forward
                    assert (self.mask == 0).sum() + (self.mask == 1).sum() == self.mask.numel(), "self.mask contains values other than 0 or 1"
                    self.exact_size = torch.sum(self.mask)
                else:
                    self.binary_mask = torch.round(self.mask)
                   
                
                x_prune = self.mask * x


                count_ones = torch.sum(torch.eq(self.mask, 1)).item()

                self.sp_ratio = count_ones/self.mask.numel()  
                self.current_ratio = torch.tensor(self.sp_ratio)
                
      
        return x_prune    
    
    def forward_prune_prune_2bit(self, x):
        
        x_int = self.to_integer_forward(x, n_bits = 2, fixed_scale=self.scale_2bit)
        real_2 = (self.scale_2bit) * (x_int - self.zero_point_2bit)
                
        return real_2
    
    def forward_prune_prune(self, x):
        
        if self.n_bits == 8:
            x_int = self.to_integer_forward(x, n_bits = self.n_bits, fixed_scale=self.scale)
            real_value = (self.scale) * (x_int - self.zero_point)
        elif self.n_bits == 4:
            x_int = self.to_integer_forward(x, n_bits = self.n_bits, fixed_scale=self.scale_4bit)
            real_value = (self.scale_4bit) * (x_int - self.zero_point_4bit)
        elif self.n_bits == 2:
            x_int = self.to_integer_forward(x, n_bits = self.n_bits, fixed_scale=self.scale_2bit)
            real_value = (self.scale_2bit) * (x_int - self.zero_point_2bit)
                
            
        return real_value
    
    
    def forward_prune(self, x):
        
        x_int = self.to_integer_forward(x, n_bits = 2, fixed_scale=self.scale_2bit)
        real_2 = (self.scale_2bit) * (x_int - self.zero_point_2bit)
                
        if self.training:
            self.current_steps += 1
            
        return [0., 0., real_2, 0., 0., 0., 0., 0., 0.]

            

QMethodMap = namedtuple('QMethodMap', ['value', 'cls'])


class PMethods(Enum):
    symmetric_uniform = QMethodMap(0, SymmetricUniformPruner)
    asymmetric_uniform = QMethodMap(1, AsymmetricUniformPruner)
    Bayesian_uniform = QMethodMap(2, ParentPruner)
    
    @property
    def cls(self):
        return self.value.cls

    @classmethod
    def list(cls):
        return [m.name for m in cls]


class PrunerNotInitializedError(Exception):
    """Raised when a pruner has not initialized"""

    def __init__(self):
        super(PrunerNotInitializedError, self).__init__('Pruner has not been initialized yet')
