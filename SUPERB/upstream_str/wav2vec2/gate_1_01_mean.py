"""Implementation of the hard Concrete distribution.

Originally from:
https://github.com/asappresearch/flop/blob/master/flop/hardconcrete.py

"""

import math

import torch
import torch.nn as nn
from typing import Optional


class Gate(nn.Module):
    """A HarcConcrete module.
    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization.

    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode, e.g.:

    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()
    """

    def __init__(
        self,
        n_in: int,
        channel_score: Optional[torch.Tensor] = None,  
        gate_threshold: float = 1e-5,
    ) -> None:
        """Initialize the HardConcrete module.
        Parameters
        ----------
        n_in : int
            The number of hard concrete variables in this mask.
        init_mean : float, optional
            Initial drop rate for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        temperature : float, optional
            Temperature used to control the sharpness of the
            distribution, by default 1.0
        stretch : float, optional
            Stretch the sampled value from [0, 1] to the interval
            [-stretch, 1 + stretch], by default 0.1.
        """
        super().__init__()

        self.n_in = n_in
        
        # self.eta_min = 0.01
        # self.t0 = 0.5
        # self.eta_max = 1  
        # self.eta_min = 0.1
        # self.eta_max = 10 
        # self.eta_min = 1
        self.reverse_tau = None
        self.eta_max = 100
        self.eta_min = 50
        self.init_sparsity = None
        self.n0 = 0.
        self.current_steps = 0.
        self.total_steps = 0.
        self.reset_threshold = False
    
        self.channel_score = channel_score
        self.compiled_mask = None
        self.gate_threshold = nn.Parameter(
            torch.tensor(gate_threshold, device='cuda'), 
            requires_grad=True 
        )
        self.reset_parameters()

    def get_channel_score(self, channel_score):
        self.channel_score = channel_score
        
    def reset_parameters(self):
        """Reset the parameters of this module."""
        self.compiled_mask = None
        # mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        # self.log_alpha.data.normal_(mean, self.init_std)

    def num(self) -> torch.Tensor:
        """Compute the expected L0 norm of this mask.
        Returns
        -------
        torch.Tensor
            The expected L0 norm.
        """
        return self.mask.sum()

    def forward(self) -> torch.Tensor:
        """Sample a hard concrete mask.
        Returns
        -------
        torch.Tensor
            The sampled binary mask
        """
        
        if self.training:
            # self.tau = (self.eta_min + 0.5 * (self.t0 - self.eta_min) * (1 + math.cos(math.pi * self.current_steps / self.total_steps)))
            if self.eta_max == self.eta_min:
                self.tau = self.eta_max
            else:
                if not self.reverse_tau:
                    if self.current_steps / self.total_steps <= 0.30:
                        self.tau = (self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.current_steps / int(self.total_steps*0.30))))
                    else:
                        self.tau = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 - math.cos(math.pi * int(self.current_steps-0.30*self.total_steps) / int(self.total_steps*0.70)))
                else:
                    if self.current_steps / self.total_steps <= 0.30:
                        self.tau = (self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 - math.cos(math.pi * self.current_steps / int(self.total_steps*0.30))))
                    else:
                        self.tau = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * int(self.current_steps-0.30*self.total_steps) / int(self.total_steps*0.70)))
            # print('self.gate_threshold', self.gate_threshold)
            # Reset the compiled mask
            if not self.reset_threshold:
                if not self.init_sparsity:
                    device = self.gate_threshold.device

                    new_value = torch.sqrt(self.channel_score.min()).clone().detach().to(device)

                    self.gate_threshold.data.copy_(new_value)

                    if self.gate_threshold.grad is not None:
                        self.gate_threshold.grad.zero_() 
                    print('init set:', new_value**2)
                    self.reset_threshold = True
                else:
                    print('init sparsity:', self.init_sparsity)
                    
                    device = self.gate_threshold.device
                    
                    sorted_scores, _ = torch.sort(self.channel_score)
                    num_scores = len(sorted_scores)

                    mid_idx = num_scores // 2

                    if num_scores % 2 == 0:
                        mid_value = (sorted_scores[mid_idx - 1] + sorted_scores[mid_idx]) / 2
                    else:
                        mid_value = sorted_scores[mid_idx]
                    # import pdb;pdb.set_trace()
                    new_value = (torch.sqrt(mid_value.squeeze())).clone().detach().to(device)
                    
                    self.gate_threshold.data.copy_(new_value)

                    if self.gate_threshold.grad is not None:
                        self.gate_threshold.grad.zero_() 
                    
                    print('init 0.5 sparsity threshold set:', new_value**2)
                    self.reset_threshold = True
                
            self.compiled_mask = None
            # Sample mask dynamically
            if self.channel_score is None:
                self.mask = torch.ones(3072, device='cuda')
            else:
                self.mask = torch.sigmoid((self.channel_score - self.gate_threshold**2)/self.tau)
                self.mask = (self.mask.round() - self.mask.detach() + self.mask)
                                     
            if self.current_steps % 1000 == 0:
                print('self.current_steps/self.total_steps:', self.current_steps/self.total_steps)
                print('self.tau:', self.tau)
            self.current_steps += 1
                
        else:
            # Compile new mask if not cached
            if self.compiled_mask is None:
                # Get expected sparsity
                if self.channel_score is None:
                    self.compiled_mask = torch.ones(3072, device='cuda')
                else:
                    # import pdb;pdb.set_trace()
                    target_device = self.channel_score.device
                    if self.gate_threshold.device != target_device:
                        self.gate_threshold.data = self.gate_threshold.data.to(target_device)
                    
                    if not self.reverse_tau:
                        print('not reverse self.eta_max:',self.eta_max)
                        self.compiled_mask = (torch.round(torch.sigmoid((self.channel_score - self.gate_threshold**2)/self.eta_max)) ==1.0)
                    else:
                        print('reverse self.eta_min:',self.eta_min)
                        self.compiled_mask = (torch.round(torch.sigmoid((self.channel_score - self.gate_threshold**2)/self.eta_min)) ==1.0)
                    # 2025.5.20 eta_min -> eta_max
                    # print('sp:', self.compiled_mask.sum().item()/self.compiled_mask.numel(), self.compiled_mask.numel(), self.compiled_mask.sum().item(), self.gate_threshold)
                
            self.mask = self.compiled_mask

        # import pdb;pdb.set_trace()
        return self.mask

    def extra_repr(self) -> str:
        return str(self.n_in)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())
