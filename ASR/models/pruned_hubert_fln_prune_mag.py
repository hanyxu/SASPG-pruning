# coding=utf-8
# author: Haoning XU
# modified: 2023-10-7

import math
import warnings
# import torch.nn.functional as F
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import sys 
# sys.path.append('/project_bdda7/bdda/hnxu/miniconda3/envs/prune_wav/lib/python3.8/site-packages/transformers')
# from deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
from utils.transformers_compat import torch_int_div

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    # BertLayer,
    # BertSelfAttention,
    # BertSelfOutput,
    Wav2Vec2LayerNormConvLayer,
    Wav2Vec2NoLayerNormConvLayer,
    Wav2Vec2GroupNormConvLayer,
    Wav2Vec2FeatureProjection,
    Wav2Vec2Attention,
    Wav2Vec2EncoderLayer,
    Wav2Vec2AdapterLayer,
    Wav2Vec2EncoderLayerStableLayerNorm,
    BaseModelOutput, # used: Wav2Vec2Encoder --> wav2vec2model
    # Wav2vec2BaseModelOutput # used: wav2vec2model
    # Wav2vec2BaseModelOutput does not inherit from BaseModelOutput
    # CausalLMOutput # final output
)

from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import ModuleUtilsMixin, apply_chunking_to_forward

from pruning.autoprune_utils import prune_model
from pruning.base_pruned_classes import PrunedActivation, FP32Acts
from pruning.base_pruned_model import PrunedModel, PrunedFromPretrainModel

from pruning.range_estimators import RangeEstimators, OptMethod
from utils.prune_ratio_utils import (
    encoder_attn_ff_prune_ratio,
    format_encoder_attn_ff_prune_ratio_message,
)
# from utils import _tb_advance_global_step, _tb_advance_token_counters, _tb_hist
from transformers.models.hubert.configuration_hubert import HubertConfig

# from anonymized_compression_package.pruning.straight_through import (
#     BayesianBitsPruner,
#     PACTPruner,
# )

_HIDDEN_STATES_START_POSITION = 2

def _compute_mask_indices(
    shape,
    mask_prob,
    mask_length,
    attention_mask = None,
    min_masks = 0,
):
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool_)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask

def gate_loss(rn):
    regularizer = 0.0
    for name, module in rn.named_modules():
        if isinstance(module, BayesianBitsPruner):
            regularizer += module.regularizer()
    return regularizer


class PrunedHubertNoLayerNormConvLayer(PrunedModel):
    def __init__(self, org_model, layer_id=0, config=None, **prune_params):

        super().__init__()
        self.config = config
        # self.layer_id = getattr(org_model, 'layer_id', None)

        # self.in_conv_dim = org_model.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        # self.out_conv_dim = org_model.conv_dim[self.layer_id]
        # import pdb;pdb.set_trace()
        self.conv = org_model[layer_id].conv
        
        self.activation = org_model[layer_id].activation # prune_model(org_model.activation, **prune_params)

        # if not isinstance(self.activation, nn.Module):
        #     if self.activation == F.gelu:
        #         self.activation = org_model[layer_id].activation # nn.GELU()
        #     else:
        #         raise NotImplementedError()
            

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class PrunedHubertLayerNormConvLayer(PrunedModel):
    def __init__(self, org_model, layer_id = 0, config=None, **prune_params):
        

        super().__init__()
        
        self.config = config
        
        # import pdb; pdb.set_trace()
        # self.layer_id = getattr(org_model, 'layer_id', None)
        # self.layer_id = getattr(self.config, 'layer_id', None)
        
        # self.in_conv_dim = org_model.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        # self.out_conv_dim = org_model.conv_dim[self.layer_id]
        self.conv = org_model[layer_id].conv
    
        # self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        
        self.layer_norm = org_model[layer_id].layer_norm
        self.activation = org_model[layer_id].activation


    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        hidden_states = hidden_states.transpose(-2, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class PrunedHubertGroupNormConvLayer(PrunedModel): # not used in Wav2vec2forCTC
    def __init__(self, org_model, layer_id=0, config=None, **prune_params):
        

        super().__init__()
        self.config = config
        # self.layer_id = getattr(org_model, 'layer_id', None)

        # self.in_conv_dim = org_model.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        # self.out_conv_dim = org_model.conv_dim[self.layer_id]
        self.conv = org_model[layer_id].conv
        self.layer_norm = org_model[layer_id].layer_norm
        
        self.activation = org_model[layer_id].activation
        # import pdb;pdb.set_trace()

    def forward(self, hidden_states):
        # import pdb;pdb.set_trace()
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
    
class PrunedHubertFeatureEncoder(PrunedModel):
    """Construct the features from raw audio waveform"""

    def __init__(self, org_model, config, **prune_params):
        
        super().__init__()

        self.config = config
        # import pdb;pdb.set_trace()
        if  self.config.feat_extract_norm == "group":
            # self.conv_layers = PrunedHubertGroupNormConvLayer(org_model.conv_layers[:1], **prune_params) + PrunedHubertNoLayerNormConvLayer(org_model.conv_layers[1:], **prune_params)
            conv_layers = [PrunedHubertGroupNormConvLayer(org_model.conv_layers, layer_id = 0, config=self.config,**prune_params)] + [
                PrunedHubertNoLayerNormConvLayer(org_model.conv_layers, layer_id= i + 1, config=self.config, **prune_params) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif self.config.feat_extract_norm == "layer":
            # self.conv_layers = PrunedHubertLayerNormConvLayer(org_model.conv_layers, **prune_params)
            conv_layers = [
                PrunedHubertLayerNormConvLayer(org_model.conv_layers, layer_id=i, config=self.config, **prune_params) for i in range(config.num_feat_extract_layers)
            ]   
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {org_model.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        self.conv_layers = nn.ModuleList(conv_layers)
        # import pdb;pdb.set_trace()
        self.gradient_checkpointing = False
        self._requires_grad = True
        
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]
        # import pdb;pdb.set_trace()
        # make sure hidden_states require grad for gradient_checkpointing

        if self._requires_grad and self.training: #  self.training is Flase in PTQ
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(conv_layer),
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)
                

        return hidden_states


class PrunedHubertFeatureProjection(PrunedModel):
    def __init__(self, org_model, config, **prune_params):
        
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        
        
        self.layer_norm = org_model.layer_norm
        self.projection = org_model.projection
      
        self.dropout = org_model.dropout

    def forward(self, hidden_states):
        # non-projected hidden states are needed for pruning
        # print(hidden_states.shape)
        """
        norm_hidden_states = self.layer_norm(hidden_states)
        """
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        # hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states
    
class PrunedHubertPositionalConvEmbedding(PrunedModel):
    # self.position_embeddings already exits in prune_model
    def __init__(self, org_model, **prune_params):
        

        super().__init__()
        # self.conv1 = org_model.conv
        # self.conv = prune_model(org_model.conv, **prune_params)
        
        self.conv = org_model.conv
     
        
        deepspeed_zero3_is_enabled = False
        # if is_deepspeed_zero3_enabled():
        # if deepspeed_zero3_is_enabled:
        #     raise ('deepspeed is not supported in pruning ')
        #     import deepspeed

        #     with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
        #         self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        #     deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
        #     deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        # else:
        # self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = org_model.padding # HubertSamePadLayer(org_model.num_conv_pos_embeddings)
        # m_act = org_model.activation 
        # self.activation = prune_model(m_act, **prune_params)
        self.activation = org_model.activation 
        # # no prune for now as prune-transformer does not prunes GELU/ 
        # whereas Tanh is pruned.
        if not isinstance(self.activation, nn.Module):
            if self.activation == F.gelu:
                self.activation = org_model.activation
            else:
                raise NotImplementedError()

    def forward(self, hidden_states):
        # import pdb;pdb.set_trace()
        hidden_states = hidden_states.transpose(1, 2)

        # hidden_states1 = self.conv1(hidden_states)
        # import pdb;pdb.set_trace()
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

class PrunedHubertFeedForward(PrunedModel):
    def __init__(self, org_model, config, i_layer, **prune_params):
        
        
        super().__init__()
        self.config = config
        self.intermediate_dropout = org_model.intermediate_dropout
        self.mask = None
        self.weight_out = None
        self.weight_in = None
        self.load = True
        
        self.hand_ratio = None
        self.i_layer = i_layer
        # self.full_width = torch.Size(org_model.intermediate_dense)[-1]
        # assert self.full_width == 3072
        # mask = torch.zeros((self.full_width, 1))
        
        self.intermediate_dense = prune_model(org_model.intermediate_dense, **prune_params)

        self.intermediate_act_fn = org_model.intermediate_act_fn

        if not isinstance(self.intermediate_act_fn, nn.Module):
            if self.intermediate_act_fn == F.gelu:
                self.intermediate_act_fn = org_model.intermediate_act_fn
            else:
                raise NotImplementedError()
            
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

        self.output_dense = prune_model(org_model.output_dense, **prune_params)
        self.output_dropout = org_model.output_dropout
        self.weight_in = torch.square(self.intermediate_dense.weight).sum(dim=1)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        # self.intermediate_dense.weight_pruneizer.pruner.pruner.use_distil = True
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

class PrunedHubertAttention(PrunedModel):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, org_model, config, i_layer, **prune_params):
        
        
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.embed_dim = org_model.embed_dim
        self.num_heads = org_model.num_heads
        self.dropout = org_model.dropout
        self.head_dim = self.embed_dim // max(int(self.num_heads), 1)

        self.hand_ratio = None
        self.i_layer = i_layer

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {org_model.num_heads})."
            )
        # self.scaling = self.head_dim**-0.5
        self.scaling = org_model.scaling
        self.is_decoder = org_model.is_decoder

        # pruned modules
        self.q_proj = prune_model(org_model.q_proj, **prune_params)
        self.k_proj = prune_model(org_model.k_proj, **prune_params)
        self.v_proj = prune_model(org_model.v_proj, **prune_params)
        
        prune_params['is_out_proj'] = True
        self.out_proj = prune_model(org_model.out_proj, **prune_params)
        # import pdb;pdb.set_trace()
        prune_params['is_out_proj'] = False
        prune_params.pop('is_out_proj')

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states = None,
        past_key_value = None,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
    ):
        
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        # print('get query proj')
        
        
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # print('get key proj')
        # print('get value proj')
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # print('get key proj')
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            # print('get value proj')
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
        #     # raise NotImplementedError('current branch of computation is not yet supported')
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        # pruner
        # MACs: 14400/768/12

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        
        # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)


        # attn_probs = nn.Dropout(p=self.dropout)(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        # print('get out proj')
        
        attn_output = self.out_proj(attn_output)
        # self.out_proj.weight_pruneizer.pruner.pruner.use_distil = True
        # _tb_advance_global_step(self)
        return attn_output, attn_weights_reshaped, past_key_value

class PrunedHubertEncoderLayer(PrunedModel):
   
    def __init__(self, org_model, config,  i_layer, **prune_params):
        
        super().__init__()
        self.config = config
        # import pdb;pdb.set_trace() 


        self.attention = PrunedHubertAttention(org_model.attention, self.config, i_layer, **prune_params) 
        # import pdb;pdb.set_trace()
        self.layer_norm = org_model.layer_norm
        self.feed_forward = PrunedHubertFeedForward(org_model.feed_forward, self.config, i_layer, **prune_params)
        self.final_layer_norm = org_model.final_layer_norm
            
        self.dropout = org_model.dropout
        # self.res3_act_pruneizer = PrunedActivation(**prune_params)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
    
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)

        
        # self.final_layer_norm.weight_pruneizer.pruner.pruner.use_distil = True
        hidden_states = self.final_layer_norm(hidden_states)
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
            # outputs = self.res3_act_pruneizer(outputs)

        return outputs

class PrunedHubertEncoder(PrunedModel): 
    def __init__(self, org_model, config, **prune_params):
        
        super().__init__()

        # self.config = config
        self.config = config
        
        self.pos_conv_embed = PrunedHubertPositionalConvEmbedding(org_model.pos_conv_embed, **prune_params)
        # self.layer_norm = prune_model(org_model.layer_norm,**prune_params)
        # self.sum_pos_embd_act_pruneizer = PrunedActivation(**prune_params)
                    
        # self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # self.layers = prune_model(
        #     org_model.layers,  **prune_params
        # )

        self.layers = nn.ModuleList([PrunedHubertEncoderLayer(org_model.layers[i], self.config, i, **prune_params) for i in range(self.config.num_hidden_layers)])
        self.dropout = org_model.dropout
        # self.res2_act_pruneizer = PrunedActivation(**prune_params)
        # self.res3_act_pruneizer = PrunedActivation(**prune_params)
        # self.res4_act_pruneizer = PrunedActivation(**prune_params)
        self.gradient_checkpointing = False # not used in PrunedBert

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
        # import pdb;pdb.set_trace()
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = False # is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 漏掉量化了 这个模块没用到
                # all_hidden_states = self.res2_act_pruneizer(all_hidden_states)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            # skip_the_layer = False
            # if skip_the_layer == True:
            #     print(dropout_probability)
            # assert skip_the_layer is False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            
            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # all_self_attentions = self.res3_act_pruneizer(all_self_attentions)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # all_hidden_states = self.res4_act_pruneizer(all_hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    
class PrunedHubertAdapterLayer(PrunedModel):
    def __init__(self, org_model, **prune_params):
        
        super().__init__()
        
        self.conv = org_model.conv
       
        # self.conv = org_model.conv
    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states

class PrunedHubertAdapter(PrunedModel):
    def __init__(self, org_model, **prune_params):
        super().__init__()
        
        self.config = org_model.config
        # feature dim might need to be down-projected
        # import pdb;pdb.set_trace()
        
        # if org_model.proj is not None:
        #     self.proj = prune_model(org_model.proj, **prune_params)
        # else:
        #     self.proj = None

        # if org_model.proj_layer_norm is not None:
        #     self.proj_layer_norm = prune_model(org_model.proj_layer_norm, **prune_params)
        # else:
        #     self.proj_layer_norm = None

        if self.config.output_hidden_size != self.config.hidden_size:
            self.proj = prune_model(org_model.proj, **prune_params)
            self.proj_layer_norm = org_model.proj_layer_norm
        else:
            self.proj = self.proj_layer_norm = None

        # self.layers = nn.ModuleList(HubertAdapterLayer(config) for _ in range(config.num_adapter_layers))
        self.layers = nn.ModuleList(PrunedHubertAdapterLayer(org_model[i].layers, ) for i in range(self.config.num_adapter_layers))
        self.layerdrop = org_model.layerdrop

    def forward(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states
    
class PrunedHubertEncoderLayerStableLayerNorm(PrunedModel):
    def __init__(self, org_model, config, i_layer, **prune_params):
        super().__init__()
        
        # import pdb;pdb.set_trace()
        # self.config = org_model.config
        self.config = config
        self.attention = PrunedHubertAttention(org_model.attention, config, i_layer, **prune_params)
        self.dropout = org_model.dropout
        self.layer_norm = org_model.layer_norm
        self.feed_forward = PrunedHubertFeedForward(org_model.feed_forward, config, i_layer, **prune_params)
        self.final_layer_norm = org_model.final_layer_norm


    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
            # outputs = self.res3_act_pruneizer(outputs)
        return outputs
    
class PrunedHubertEncoderStableLayerNorm(PrunedModel):
    def __init__(self, org_model, config, **prune_params):
        
        super().__init__()

        # self.config = config
        self.config = org_model.config
        self.no_skip_first_round = False
        self.first_round = True
        self.sub_size_loss = 0.  
        self.sub_size_loss_list = []      
        self.num_iter = 0
        
        # self.config = org_model.config
        self.pos_conv_embed = PrunedHubertPositionalConvEmbedding(org_model.pos_conv_embed, **prune_params)
        
        self.layer_norm = org_model.layer_norm
       
        # self.layers = nn.ModuleList([PrunedHubertEncoderLayerStableLayerNorm(org_model[i].layers, **prune_params) for i in range(self.config.num_hidden_layers)])
        # self.layers = prune_model(org_model.layers, specials={HubertEncoderLayerStableLayerNorm: PrunedHubertEncoderLayerStableLayerNorm}, **prune_params)
        # import pdb;pdb.set_trace()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([PrunedHubertEncoderLayerStableLayerNorm(org_model.layers[i], self.config, i, **prune_params) for i in range(self.config.num_hidden_layers)])

        self.dropout = org_model.dropout
        

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings

        hidden_states = self.dropout(hidden_states)


        # deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        deepspeed_zero3_is_enabled = False # is_deepspeed_zero3_enabled()
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # all_hidden_states = self.res2_act_pruneizer(all_hidden_states)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            # skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) and self.no_skip_first_round else False
            skip_the_layer = False
            # print('self.num', self.num_iter%24, skip_the_layer)
            # self.num_iter += 1
            if not skip_the_layer or deepspeed_zero3_is_enabled:
            #     # under deepspeed zero3 all gpus must run in sync
            #     # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    
                        # self.sub_size_loss += layer.get_gate_loss_direct()
                        # print('self.sub_size_loss',layer.get_gate_loss_direct())
                        # import pdb;pdb.set_trace()
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                    if not self.first_round:
                        # self.sub_size_loss += layer.get_gate_loss_direct()
                        # print('self.sub_size_loss',layer.get_gate_loss_direct())
                        self.sub_size_loss_list.append(layer.get_gate_loss_direct())
                    # import pdb;pdb.set_trace()
                else:
                        # self.sub_size_loss += layer.get_gate_loss_direct()
                        # print('self.sub_size_loss',layer.get_gate_loss_direct())
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                    # if self.training and not self.first_round:
                    #     # self.sub_size_loss_list.append(layer.get_gate_loss_direct())
                    #     self.sub_size_loss += layer.get_gate_loss_direct()
                    #     print('self.sub_size_loss',layer.get_gate_loss_direct())
                        
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # all_self_attentions = self.res3_act_pruneizer(all_self_attentions)
                
        if self.first_round == True and self.training:
            self.first_round = False
            
        if self.no_skip_first_round == False and self.training:
            self.no_skip_first_round = True
            
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # all_hidden_states = self.res4_act_pruneizer(all_hidden_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return {'last_hidden_state':hidden_states,
            "hidden_states":all_hidden_states,
            "attentions":all_self_attentions,
            # "size_loss":self.sub_size_loss
            "size_loss":self.sub_size_loss_list
        }

class PrunedHubertPreTrainedModel(PrunedFromPretrainModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HubertConfig
    base_model_prefix = "hubert"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            if is_deepspeed_zero3_enabled():
                import deepspeed

                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                nn.init.kaiming_normal_(module.weight.data)

        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PrunedHubertEncoder, PrunedHubertEncoderStableLayerNorm)):
            module.gradient_checkpointing = value

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
            
class PrunedHubertModel(PrunedHubertPreTrainedModel):
    def __init__(self, org_model,  **prune_params):
        super().__init__(org_model.config)
        # self.config = config
        # import pdb;pdb.set_trace()
        self.config = org_model.config
        
        # import pdb;pdb.set_trace()
        self.feature_extractor = PrunedHubertFeatureEncoder(org_model.feature_extractor, self.config, **prune_params)
        self.feature_projection = PrunedHubertFeatureProjection(org_model.feature_projection, self.config, **prune_params)

        # model only needs masking vector if mask prob is > 0.0
        if self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(self.config.hidden_size).uniform_())
            # import pdb;pdb.set_trace()
            # self.masked_spec_embed = nn.Parameter(torch.FloatTensor(self.config.hidden_size))

        if self.config.do_stable_layer_norm:
            # raise NotImplementedError('current branch of computation is not yet supported')
            # not used
            self.encoder = PrunedHubertEncoderStableLayerNorm(org_model.encoder, self.config, **prune_params)
        else:
            self.encoder = PrunedHubertEncoder(org_model.encoder, self.config, **prune_params)

        # import pdb;pdb.set_trace()
        # self.adapter = PrunedHubertAdapter(org_model.adapter, **prune_params) if config.add_adapter else None
        # import pdb;pdb.set_trace()

        # if self.config.add_adapter is True:
        #     self.adapter = PrunedHubertAdapter(org_model.adapter, self.config, **prune_params)
        # else:
        #     self.adapter = None
        # Initialize weights and apply final processing
        # self.post_init()
        # self.get_feat_extract_output_lengths = org_model._get_feat_extract_output_lengths
        
        # self.res1_act_pruneizer = PrunedActivation(**prune_params)
    # def set_precision_level(self, precison):
    #     self.n_bits = precison
    #     self.n_bits_act = precison

    def _get_feat_extract_output_lengths(
        self, input_lengths, add_adapter = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        # add_adapter = self.config.add_adapter if add_adapter is None else add_adapter # no adapter in hubert

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask
    
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states,
        mask_time_indices = None,
        attention_mask = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # import pdb;pdb.set_trace()
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # import pdb;pdb.set_trace()
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states


    def forward(
        self,
        input_values ,
        attention_mask = None,
        mask_time_indices = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        # import pdb;pdb.set_trace()
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask
            )
        """
        hidden_states, extract_features = self.feature_projection(extract_features)
        # import pdb;pdb.set_trace()
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )
        """
        hidden_states = self.feature_projection(extract_features)
        # import pdb;pdb.set_trace()
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs['last_hidden_state']

        # if self.adapter is not None:
        #     hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return {"last_hidden_state":hidden_states,
            "hidden_states":encoder_outputs["hidden_states"],
            "attentions":encoder_outputs["attentions"],
            "size_loss":encoder_outputs["size_loss"]}
        
    
class PrunedHubertForCTC(PrunedHubertPreTrainedModel):
    def __init__(self, org_model, prune_setup=None, precision_levels=[8,4], **prune_params):
        
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        
        super().__init__(org_model.config)
        
        # self.layer_id = getattr(org_model, 'layer_id', None)

        # self.in_conv_dim = org_model.conv_dim[self.layer_id - 1] if self.layer_id > 0 else 1
        # self.out_conv_dim = org_model.conv_dim[self.layer_id]
        # import pdb;pdb.set_trace()
        self.precision_levels = precision_levels
        # self.width_levels = [30, 23]
        # self.prec_levels = [8, 4]
        # self.depth_levels = [12, 10]
        self.config = org_model.config
        
        self.times = 0
        self.calib = 0
        
        self.all_pruned = prune_params['weight_prune'] and prune_params['act_prune']
        self.width = None
        self.prec = None
        self.depth = 24
                
        self.lmb = prune_params['lmb']
        self.lmb_dis = prune_params['lmb_dis']
        self.idx = 0
        self.single_bit = int(prune_params['single_bit'])
        prune_params.pop('single_bit')
        
        self.fix_prob = prune_params['fix_prob']
        prune_params.pop('fix_prob')
        self.mag_prune = prune_params['mag_prune']
        prune_params.pop('mag_prune')
        prune_params.pop('hand_ratio')
        
        prune_params.pop('hard')
        prune_params.pop('only_size_hard')
        prune_params.pop('decay_tau')
        prune_params.pop('nasp_ladder', None)
        
        prune_params.pop('weight_prune')
        prune_params.pop('act_prune')
        prune_params.pop('lmb')
        prune_params.pop('lmb_dis')

        prune_params.pop('is_arc_prune')
        prune_params.pop('prune_2_4')
        
        self.max_bit = prune_params['max_bit']
        self.min_bit = prune_params['min_bit']
        prune_params.pop('max_bit')
        prune_params.pop('min_bit')
        
        self.max_prune_ratio = prune_params['max_prune_ratio']
        self.min_prune_ratio = prune_params['min_prune_ratio']
        prune_params.pop('max_prune_ratio')
        prune_params.pop('min_prune_ratio')
        
        self.model_path = False
        self.thre_model_path = False
        self.sum_shape = 1024*1024*4*24 + 1024*4096*2*24 + 1024*3*24 + 4096*24
        # unstr wav2vec2没有加bias
        self.first_round = True
        
        if 'model_path' in prune_params:
            self.load_path = prune_params['model_path']
            prune_params.pop('model_path')
            self.weight_dict = torch.load(self.load_path)
            self.delete_dict()
            self.clean_dict()
            self.model_path = True
        elif 'thre_model_path' in prune_params:
            self.load_path = prune_params['thre_model_path']
            prune_params.pop('thre_model_path')
            self.weight_dict = torch.load(self.load_path)
            # self.delete_dict()
            # self.clean_dict()
            self.thre_model_path = True
        
        self.freeze_param = 0
        self.freeze_thre = 0
        
        self.exact_size = 0
        self.current_steps = 0
        
        # import pdb;pdb.set_trace()
        self.reg_type = prune_params['reg_type']
        self.avg_loss = 0
        self.avg_ce_loss = 0
        self.avg_c_loss = 0

        self.save_path = prune_params['save_path']
        prune_params.pop('save_path')
        
        self.hubert = PrunedHubertModel(org_model.hubert, **prune_params)
        # import pdb; pdb.set_trace()
        # self.dropout = nn.Dropout(config.final_dropout)
        if hasattr(org_model, 'dropout'):
            self.dropout = org_model.dropout
            # self.dropout_low = org_model.dropout
        
        if self.config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            self.config.output_hidden_size if hasattr(self.config, "add_adapter") and self.config.add_adapter else self.config.hidden_size
        )
        # self.get_feat_extract_output_lengths = org_model._get_feat_extract_output_lengths
        prune_params_ = prune_params.copy()

        # self.lm_head = org_model.lm_head
        # import pdb;pdb.set_trace()
        
       
        self.lm_head = org_model.lm_head

        
        # Initialize weights and apply final processing
        # self.post_init()
        if self.model_path and self.fix_prob and self._weight_dict_is_structural_pruned_mag(
            self.weight_dict
        ):
            _sd = {k: v for k, v in self.weight_dict.items() if "gate_threshold" not in k}
            self.load_state_dict(_sd, strict=False)
            self.model_path = False
            self.weight_dict = {}

    @staticmethod
    def _weight_dict_is_structural_pruned_mag(state_dict):
        if not state_dict:
            return False
        return not any(
            "weight_pruneizer_saspg" in k and k.endswith("threshold_2") for k in state_dict
        )

    def delete_dict(self):
        keys_to_delete = []

        for key in self.weight_dict.keys():
            if not any(substring in key for substring in ['delta', 'zero', 'signed', 'threshold']):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.weight_dict[key]
        

    def clean_dict(self):
        keys_to_delete = []
        index = 0
        for key in self.weight_dict.keys():
            if 'range_estimator' in key:
                index = 1
        
        if index == 1:
            for key in self.weight_dict.keys():
                if 'range_estimator' not in key:
                    base_key = '.'.join(key.split('.')[:-3])
                    pruner_key = '.'.join(key.split('.')[-3:])
                    assert self.weight_dict[base_key + '.range_estimator.' + pruner_key] == self.weight_dict[key]
                    keys_to_delete.append(base_key + '.range_estimator.' + pruner_key)
                    assert self.weight_dict[base_key + '.range_estimator_2.' + pruner_key] == self.weight_dict[key]
                    keys_to_delete.append(base_key + '.range_estimator_2.' + pruner_key)
                    assert self.weight_dict[base_key + '.range_estimator_4.' + pruner_key] == self.weight_dict[key]
                    keys_to_delete.append(base_key + '.range_estimator_4.' + pruner_key)
                # elif 'range_estimator' not in key and 'sign' in key:
                #     base_key = '.'.join(key.split('.')[:-3])
                #     pruner_key = '.'.join(key.split('.')[-3:])
                #     assert self.weight_dict[base_key + '.range_estimator.' + pruner_key] == self.weight_dict[key]
                #     keys_to_delete.append(base_key + '.range_estimator.' + pruner_key)
                #     assert self.weight_dict[base_key + '.range_estimator_2.' + pruner_key] == self.weight_dict[key]
                #     keys_to_delete.append(base_key + '.range_estimator_2.' + pruner_key)
                #     assert self.weight_dict[base_key + '.range_estimator_4.' + pruner_key] == self.weight_dict[key]
                #     keys_to_delete.append(base_key + '.range_estimator_4.' + pruner_key)
            for key in keys_to_delete:
                del self.weight_dict[key]

        

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    def _get_feat_extract_output_lengths(
        self, input_lengths, add_adapter = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        # add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths
    
    # def get_gate_loss(self):
    #     regularizer = 0.0
    #     for name, module in self.named_modules():
    #         import pdb;pdb.set_trace()
    #         if hasattr(self.pruner, 'pruner'):
    #                 regularizer += self.pruner.pruner.regularizer()
    #     import pdb;pdb.set_trace()
    #     return regularizer
    
    def forward(
        self,
        input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states  = None,
        return_dict  = None,
        labels = None,
        prec = None,
        width = None,
        depth = None,
    ):
        assert len(self.precision_levels) > 0
        # print(self.precision_levels)
        
        """if not self.training and self.calib < self.num_acts: # forward 2 bit to get the initial value 
            print('calibrate ... ')
            
            loss = []
            logits =[]
            # self.hubert.set_precision_level_mask(23)
            # self.lm_head.set_precision_level_mask(23)
                
            for precision in self.precision_levels:
                if precision != 8:
                    assert precision == 4 or precision == 2 or precision == 3
                self.hubert.set_precision_level_direct(precision)
                # self.lm_head.set_precision_level_direct(precision)
                
                # self.hubert.set_precision_level_mask(30)
                # self.lm_head.set_precision_level_mask(30)
                
                # import pdb;pdb.set_trace()
                self.hubert.return_q()
                CausalLMOutput_multi = self.subforward(
                    input_values,
                    attention_mask,
                    output_attentions,
                    output_hidden_states,
                    return_dict,
                    labels,
                    )
                # if len(self.precision_levels) == 1:
                #     return {
                #     'loss': CausalLMOutput_multi['loss'],
                #     'loss0': 0,
                #     'loss1': 0,
                #     'logits': CausalLMOutput_multi['logits'],
                #     'hidden_states': None,
                #     'attentions': None
                #         }
                logits.append(CausalLMOutput_multi['logits'])
                loss.append(CausalLMOutput_multi['loss'])

            self.loss = loss[0] + loss[1] + loss[2] # + F.kl_div(log_probs_4bit, probs_8bit.detach(), 
            self.calib += 1
            print('calibrate done ... ')
            
            """
            
                        
        if self.model_path:
            index_delta = 0
            
            for name, module in self.named_modules():
                if name.endswith('weight_pruneizer'): # or name.endswith('activation_pruner'):
                    # print(name)  # 打印模块的名称
                    # print(name, self.weight_dict[f'{name}.pruner.pruner._delta_2bit'],  module.pruner.pruner._delta_2bit)
                    # 之前是不加.data, 2024-7-2加上.data 是否保存梯度
                    if index_delta == 1:
                        _d = f"{name}.pruner.pruner._delta"
                        if _d in self.weight_dict:
                            module.pruner.pruner._delta.data = self.weight_dict[_d]
                            module.pruner.pruner._delta_2bit.data = self.weight_dict[
                                f"{name}.pruner.pruner._delta_2bit"
                            ]
                            module.pruner.pruner._delta_4bit.data = self.weight_dict[
                                f"{name}.pruner.pruner._delta_4bit"
                            ]
                            module.pruner.pruner._signed.data = self.weight_dict[
                                f"{name}.pruner.pruner._signed"
                            ]
                    
                    if not self.thre_model_path and self.fix_prob:
                        for _suf in ("threshold_2", "threshold_4", "threshold_8"):
                            _k = f"{name}.pruner.pruner.{_suf}"
                            if _k in self.weight_dict:
                                getattr(module.pruner.pruner, _suf).data = self.weight_dict[_k]
                    # module.range_estimator.pruner.pruner._delta = self.weight_dict[f'{name}.range_estimator.pruner.pruner._delta']
                    # module.range_estimator_2.pruner.pruner._delta = self.weight_dict[f'{name}.range_estimator_2.pruner.pruner._delta']
                    # module.range_estimator_4.pruner.pruner._delta = self.weight_dict[f'{name}.range_estimator_4.pruner.pruner._delta']
                    
                    # module.range_estimator_2.pruner.pruner._delta_2bit = self.weight_dict[f'{name}.range_estimator_2.pruner.pruner._delta_2bit']
                    # module.range_estimator.pruner.pruner._delta_2bit = self.weight_dict[f'{name}.range_estimator.pruner.pruner._delta_2bit']
                    # module.range_estimator_4.pruner.pruner._delta_2bit = self.weight_dict[f'{name}.range_estimator_4.pruner.pruner._delta_2bit']
                    
                    # module.range_estimator.pruner.pruner._delta_4bit = self.weight_dict[f'{name}.range_estimator.pruner.pruner._delta_4bit']
                    # module.range_estimator_2.pruner.pruner._delta_4bit = self.weight_dict[f'{name}.range_estimator_2.pruner.pruner._delta_4bit']
                    # module.range_estimator_4.pruner.pruner._delta_4bit = self.weight_dict[f'{name}.range_estimator_4.pruner.pruner._delta_4bit']
                    
                    # module.range_estimator.pruner.pruner._signed = self.weight_dict[f'{name}.range_estimator.pruner.pruner._signed']
                    # module.range_estimator_2.pruner.pruner._signed = self.weight_dict[f'{name}.range_estimator_2.pruner.pruner._signed']
                    # module.range_estimator_4.pruner.pruner._signed = self.weight_dict[f'{name}.range_estimator_4.pruner.pruner._signed']
        # for k,v in self.weight_dict.items():
        #     print(k,v)
        # for name, module in self.named_modules():
        #     if name.endswith('weight_pruneizer'): # or name.endswith('activation_pruner'):
        #         # print(name)  # 打印模块的名称
        #         print(name,  module.pruner.pruner._delta_2bit)
            self.model_path = False
                
        if self.thre_model_path:
            for name, module in self.named_modules():
                if name.endswith('weight_pruneizer_saspg'): # or name.endswith('activation_pruner'):
                    # print(name)  # 打印模块的名称
                    _k = f'{name}.pruner.pruner.current_ratio'
                    if _k in self.weight_dict:
                        module.pruner.pruner.current_ratio.data = self.weight_dict[_k]
                        print(name, self.weight_dict[_k],  module.pruner.pruner.current_ratio)
                    
            self.thre_model_path = False
                    
        """return {
            'loss': self.loss,
            'loss0': loss[0],
            'loss1': loss[1],
            'loss2': loss[2],
            'logits': (logits[0], logits[1], logits[2]),
            'hidden_states': None,
            'attentions': None
                }"""
        

        
        loss_list = []
        logits_list =[]
        log_softmax_list = []

        hidden_states_list = []

        # for name, module in self.named_modules():
        #     if name.endswith('weight_pruneizer'): # or name.endswith('activation_pruner'):
        #         # print(name)  # 打印模块的名称
        #         print(name,  self.weight_dict[f'{name}.pruner.pruner._delta_2bit'], module.pruner.pruner._delta_2bit)
        #         print(name,  self.weight_dict[f'{name}.pruner.pruner._delta_4bit'], module.pruner.pruner._delta_4bit)
        
        self.hubert.return_mask()
        # self.lm_head.return_q()
        CausalLMOutput_multi = self.subforward(
            input_values,
            attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            labels,
            depth=self.depth)
        logits_list.append(CausalLMOutput_multi['logits'])
        loss_list.append(CausalLMOutput_multi['loss'])
        log_softmax_list.append(CausalLMOutput_multi['log_softmax'])
        hidden_states_list.append(CausalLMOutput_multi['hidden_states_list'])
        
        # if self.training:
        #     softmax_logits_mq = F.softmax(logits_list[3], dim=-1, dtype=torch.float32).transpose(0, 1)
        
        # self.hubert.return_q8()
        # # self.lm_head.return_q8()
        # CausalLMOutput_multi = self.subforward(
        #     input_values,
        #     attention_mask,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        #     labels,
        #     depth=self.depth)
        # logits_list.append(CausalLMOutput_multi['logits'])
        # loss_list.append(CausalLMOutput_multi['loss'])
        # log_softmax_list.append(CausalLMOutput_multi['log_softmax'])
        # hidden_states_list.append(CausalLMOutput_multi['hidden_states_list'])
            
        # if self.training:
        #     log_probs_q8 = hidden_states_list[4]
            
            # if self.reg_type == 'distilctc':
            #     for i in range(11):
            #         distil_loss += F.mse_loss(log_probs_q8[i], probs_fp[i].detach(), reduction='sum')
        
        # import pdb;pdb.set_trace()
        """
        self.hubert.set_precision_level_direct(precision)
        self.lm_head.set_precision_level_direct(precision)
        """
        
        ctc_fp_loss = 0.
        ctc_q_loss = 0.
        size_loss = 0.
        KL_loss = 0.
        cos_loss = 0.

        KL_fp_8 = 0.
        KL_fp_4 = 0.
        KL_fp_2 = 0.
        KL_fp_mix = 0.

            
        ctc_8_loss, ctc_4_loss, ctc_2_loss = 0, 0, 0
        if loss_list[0] is None:
            self.loss = None
        else:
            ctc_q_loss = loss_list[0]
        # print('logits_list[1].equal(logits_list[2])',logits_list[1].equal(logits_list[2]))
        # assert not logits_list[1].equal(logits_list[2])
        # assert not logits_list[0].equal(logits_list[2])
        # assert not logits_list[1].equal(logits_list[0])
        # self.get_gate_loss
        # import pdb;pdb.set_trace()
        size_loss = self.hubert.get_gate_loss_prune()
        
        # if self.first_round:
        #     self.hubert.get_gate_loss_direct()
        #     print('size_loss all',self.hubert.get_gate_loss_direct())
        #     self.first_round = False
        # else:
        #     if isinstance(self.size_loss_all, list):
        #         size_loss = sum(self.size_loss_all)
        #     else:
        #         size_loss = self.size_loss_all
            # print('size_loss my',size_loss)
            
        # print('size_loss',size_loss)
        if not isinstance(size_loss, float):
            size_loss = size_loss.to('cuda')
        # KL_loss = self.hubert.get_gate_loss_KL()
        
        if self.reg_type == 'disweight':
            distil_loss = self.hubert.get_gate_loss_disweight()
        elif self.reg_type == 'disweightout':
            distil_loss = self.hubert.get_gate_loss_disweightout()
        
        self.exact_size = self.hubert.get_exact_size_prune()
        self.exact_prune_ratio = encoder_attn_ff_prune_ratio(
            self.exact_size, self.sum_shape
        )
        # print('self.exact_bit',self.exact_bit)
        # threshold_all = self.hubert.get_threshold_all()
        
        # cos_loss = self.hubert.get_gate_loss_cos()
        
        
        mixed_precision_list = [i for i in self.hubert.get_mixed_sparsity_fix()]
        
        pruned_list = list(filter(None, mixed_precision_list))
                
        timestamp = datetime.now().strftime('%Y-%m-%d-%H.%M')
        
        timestamp = timestamp[:-1]
        
        remove_ln_list = []
        
        for i in pruned_list:
            # if len(list(i.values())[0]) != 1: # do not output Layernorm params [768]
            remove_ln_list.append(i)
        
        # import pdb;pdb.set_trace()
        mixed_prec_path = os.path.join(self.save_path, 'mixed_sparsity')
        if not os.path.exists(mixed_prec_path):
            os.makedirs(mixed_prec_path)
    
        if self.training:
            file_name = mixed_prec_path + '/train_mixed_sparsity_{}.npy'.format(timestamp)
        else:
            file_name = mixed_prec_path + '/eval_mixed_sparsity_{}.npy'.format(timestamp)
        
        if not os.path.exists(file_name):
            np.save(file_name, remove_ln_list)

        mixed_thre_list = [i for i in self.hubert.get_mixed_thre_fix()]
        
        pruned_list = list(filter(None, mixed_thre_list))
    
        
        remove_ln_list = []
        
        for i in pruned_list:
            # if len(list(i.values())[0]) != 1: # do not output Layernorm params [768]
            remove_ln_list.append(i)
        
        # import pdb;pdb.set_trace()
        mixed_prec_path = os.path.join(self.save_path, 'mixed_thre')
        if not os.path.exists(mixed_prec_path):
            os.makedirs(mixed_prec_path)
    
        if self.training:
            file_name = mixed_prec_path + '/train_mixed_thre_{}.npy'.format(timestamp)
        else:
            file_name = mixed_prec_path + '/eval_mixed_thre_{}.npy'.format(timestamp)
        
        if not os.path.exists(file_name):
            np.save(file_name, remove_ln_list)
            
                
        if not isinstance(KL_loss, float):
            KL_loss = KL_loss.to('cuda')
        else:
            KL_loss = torch.tensor(0.).to('cuda')
            
        if (self.fix_prob or self.mag_prune):
            if self.freeze_thre == 0:
                for name, param in self.named_parameters():
                    # print(name)
                    if "threshold" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        
            self.freeze_thre = 1
            size_loss = 0

        # if self.loss is not None:
            # self.loss = ctc_q_loss + ctc_fp_loss + self.lmb * size_loss + self.lmb_dis * (distil_loss + cos_loss)
            # if (self.lmb * size_loss) > (8.0 * self.lmb / 2e-7): # 7.9
                
            # ctc_2_loss = 0
            
            # print('self.exact_bit',self.exact_bit)
        if not (self.fix_prob or self.mag_prune):
            if self.exact_prune_ratio > self.max_prune_ratio:
                self.loss = ctc_q_loss + self.lmb * size_loss + self.lmb_dis * (KL_loss)
            elif self.exact_prune_ratio < self.min_prune_ratio:
                self.loss = ctc_q_loss - self.lmb * size_loss + self.lmb_dis * (KL_loss)
            elif self.exact_prune_ratio <= self.max_prune_ratio and self.exact_prune_ratio >= self.min_prune_ratio:
                self.loss = ctc_q_loss + self.lmb_dis * (KL_loss)
            else:
                raise NotImplementedError()
        else:
            self.loss = ctc_q_loss + self.lmb * size_loss + self.lmb_dis * (KL_loss)
            # else:
            #     # self.hubert.fix_threshold()
            #     if self.freeze_thre == 0:
            #         for name, param in self.named_parameters():
            #             # print(name)
            #             if "threshold" in name:
            #                 param.requires_grad = False
            #             else:
            #                 param.requires_grad = True
            #         self.freeze_thre = 1
            #     size_loss = 0
            #     self.loss = ctc_q_loss + self.lmb_dis * (distil_loss)
        
        if not self.training:
            self.current_steps += 1
            if self.current_steps % 500 == 0 and self.current_steps > 0:
                _, _pr_msg = format_encoder_attn_ff_prune_ratio_message(
                    "PrunedHubertForCTC",
                    float(self.exact_size),
                    float(self.sum_shape),
                    extra="MAG unstr dense-shape mag_mask count",
                )
                print(_pr_msg)
                
        return {
                'loss': self.loss,
                'loss1': ctc_q_loss,
                'loss2': self.lmb_dis * KL_loss,
                'loss3': self.lmb * size_loss,
                'loss4':ctc_8_loss,
                'loss5':ctc_4_loss,
                'loss6':ctc_fp_loss,
                'loss7': ctc_2_loss,
                'loss8':self.exact_prune_ratio,   
                'kl_div': 0.,
                'kl_div_1':0.,
                'kl_div_2':0.,
                'kl_div_3':self.lmb_dis * KL_fp_8,
                'kl_div_4':self.lmb_dis * KL_fp_4,
                'kl_div_5':self.lmb_dis * KL_fp_2,
                'kl_div_6':self.lmb_dis * KL_fp_mix,
                'logits': logits_list[0], # 5ctc first modified
                'hidden_states': None,
                'attentions': None
                    }
        
        # c_loss.cuda()
        # self.lmb.cuda()
        # print('^^^^^^^^^^^^^^^^')
        # print(c_loss.device)
        # print(self.loss)
        # import pdb;pdb.set_trace()
        # return {
        #     'loss': self.loss,
        #     'loss0': ctc_q_loss,
        #     'loss1': self.lmb * size_loss,
        #     'loss2': self.lmb_dis * KL_loss,
        #     'loss3':self.lmb * cos_loss,
        #     'loss4':ctc_fp_loss,
        #     'loss5':ctc_8_loss,
        #     'loss6':ctc_4_loss,
        #     'loss7':ctc_2_loss,
        #     # 'loss3': self.lmb * c_loss,
        #     # 'kl_div_1':self.avg_loss,
        #     # 'kl_div_2':self.avg_ce_loss,
        #     # 'kl_div_3':self.avg_c_loss,
        #     # 'kl_div': loss_list[2],
        #     'logits': logits_list[1] if self.training else logits_list[0], # 5ctc first modified
        #     'hidden_states': None,
        #     'attentions': None
        #         }
        # return CausalLMOutput(loss=self.loss, logits=(logits_list[0],logits_list[1]),hidden_states=None, attentions=None)
    
    def subforward(
        self,
        input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states  = None,
        return_dict  = None,
        labels = None,
        depth = 24,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # self.hubert.set_precision_level_direct(precision)
        #     # self.hubert.set_precision_level(precision)
        # self.lm_head.set_precision_level_direct(precision)

        # output_hidden_states = True
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        self.size_loss_all =  outputs["size_loss"]
        
        hidden_states = outputs['last_hidden_state']
        hidden_states = self.dropout(hidden_states)
        # import pdb;pdb.set_trace()
        logits = self.lm_head(hidden_states)
        # self.lm_head.weight_pruneizer.pruner.pruner.use_distil = True
        # import pdb;pdb.set_trace()
        
        hidden_states_list = []

        # for i in range(11):
        #     hidden_states_list.append(self.dropout(outputs[2][i+1]))
        # hidden_states_low_11 = outputs[2][11]
        # # assert hidden_states_8.shape == hidden_states.shape
        # # print(outputs[2][12].equal(outputs[0]))
        # hidden_states_low_11 = self.dropout(hidden_states_low_11)
        # # logits_low_11 = self.lm_head(hidden_states_low_11)
        
        # hidden_states_low_10 = outputs[2][10]
        # # assert hidden_states_8.shape == hidden_states.shape
        # # print(outputs[2][12].equal(outputs[0]))
        # hidden_states_low_10 = self.dropout(hidden_states_low_10)
        # # logits_low_10 = self.lm_head(hidden_states_low_10)


        loss = None
        loss_low = None
        loss_low_10 = None
        loss_low_11 = None
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            # log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            
            
            with torch.backends.cudnn.flags(enabled=False):
                if depth == 24:
                    loss = nn.functional.ctc_loss(
                        log_probs,
                        flattened_targets,
                        input_lengths,
                        target_lengths,
                        blank=0 if self.config.pad_token_id is None else self.config.pad_token_id,
                        reduction=self.config.ctc_loss_reduction,
                        zero_infinity=self.config.ctc_zero_infinity,
                    )
                # elif depth == 11:
                #     hidden_states_low_11 = outputs[2][11]
                #     # assert hidden_states_8.shape == hidden_states.shape
                #     # print(outputs[2][12].equal(outputs[0]))
                #     hidden_states_low_11 = self.dropout(hidden_states_low_11)
                #     logits_low_11 = self.lm_head(hidden_states_low_11)
                #     log_probs_low_11 = nn.functional.log_softmax(logits_low_11, dim=-1, dtype=torch.float32).transpose(0, 1)
                #     loss_low_11 = nn.functional.ctc_loss(
                #         log_probs_low_11,
                #         flattened_targets,
                #         input_lengths,
                #         target_lengths,
                #         blank=0 if self.config.pad_token_id is None else self.config.pad_token_id,
                #         reduction=self.config.ctc_loss_reduction,
                #         zero_infinity=self.config.ctc_zero_infinity,
                #     )
                # elif depth == 10:
                #     hidden_states_low_10 = outputs[2][10]
                #     # assert hidden_states_8.shape == hidden_states.shape
                #     # print(outputs[2][12].equal(outputs[0]))
                #     hidden_states_low_10 = self.dropout(hidden_states_low_10)
                #     logits_low_10 = self.lm_head(hidden_states_low_10)
                #     log_probs_low_10 = nn.functional.log_softmax(logits_low_10, dim=-1, dtype=torch.float32).transpose(0, 1)
                #     loss_low_10 = nn.functional.ctc_loss(
                #         log_probs_low_10,
                #         flattened_targets,
                #         input_lengths,
                #         target_lengths,
                #         blank=0 if self.config.pad_token_id is None else self.config.pad_token_id,
                #         reduction=self.config.ctc_loss_reduction,
                #         zero_infinity=self.config.ctc_zero_infinity,
                #     )

        if not return_dict:
            # raise NotImplementedError('current branch of computation is not yet supported')
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # import pdb;pdb.set_trace()
        # return loss
        if depth == 24:
            output_dict = {"loss":loss, "logits":logits, 'log_softmax':log_probs, 'hidden_state':outputs['hidden_states'], "attentions":outputs['attentions'], "hidden_states_list":hidden_states_list}
        # elif depth == 11:
        #     output_dict = {"loss":loss_low_11, "logits":logits_low_11, 'log_softmax':log_probs, 'hidden_state':outputs.hidden_states, "attentions":outputs.attentions, "hidden_states_list":hidden_states_list}
        # elif depth == 10:
        #     output_dict = {"loss":loss_low_10, "logits":logits_low_10, 'log_softmax':log_probs, 'hidden_state':outputs.hidden_states, "attentions":outputs.attentions, "hidden_states_list":hidden_states_list}
        else:
            raise NotImplementedError()
        
        return output_dict
        # return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
