# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

from functools import wraps

import click

from pruning.pruners_direct import PMethods
# from pruning.pruners_direct import PMethods as PMethods_direct
# from pruning.pruners_trunc import PMethods

# from pruning.pruners import PMethods as PMethods_orig
from pruning.range_estimators import RangeEstimators, OptMethod
from utils.utils import DotDict


def split_dict(src: dict, include=()):
    """
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.

    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    """
    result = DotDict()

    for arg in include:
        result[arg] = src[arg]
    remainder = {key: val for key, val in src.items() if key not in include}
    return result, remainder


def pruning_options(func):
    
    @click.option(
        '--qmethod',
        type=click.Choice(PMethods.list()),
        default="Bayesian_uniform",
        required=True,
        help='Pruning scheme to use.',
    )
    @click.option(
        '--qmethod-act',
        type=click.Choice(PMethods.list()),
        default="Bayesian_uniform",
        help='Pruning scheme for activation to use. If not specified `--qmethod` is used.',
)
    @click.option(
        '--weight-prune-method',
        # default=RangeEstimators.current_minmax.name,
        default=RangeEstimators.MSE.name, # 2024-7-4 modified.
        type=click.Choice(RangeEstimators.list()),
        help='Method to determine weight pruning clipping thresholds.',
    )
    @click.option(
        '--weight-opt-method',
        # default=OptMethod.grid.name, # only used in MSE or Entropy
        default=OptMethod.golden_section.name, # 2024-7-4 modified.
        type=click.Choice(OptMethod.list()),
        help='Optimization procedure for activation pruning clipping thresholds',
    )
    @click.option(
        '--num-candidates',
        type=int,
        default=None,
        help='Number of grid points for grid search in MSE range method.',
    )
    @click.option('--max-bit', default=3.9, type=float, help='Default number of pruning bits.')
    @click.option('--min-bit', default=3.8, type=float, help='Default number of pruning bits.')

    @click.option('--value-1', default=0., type=float, help='Default number of pruning bits.')
    @click.option('--value-075', default=0., type=float, help='Default number of pruning bits.')
    @click.option('--value-05', default=0., type=float, help='Default number of pruning bits.')
    @click.option('--value-025', default=0., type=float, help='Default number of pruning bits.')
    @click.option('--value-0125', default=0., type=float, help='Default number of pruning bits.')
    @click.option('--value-01', default=0., type=float, help='Default number of pruning bits.')
    @click.option('--value-0075', default=0., type=float, help='Default number of pruning bits.')
    
    @click.option('--max-prune-ratio', default=3.9, type=float, help='Default number of pruning bits.')
    @click.option('--min-prune-ratio', default=3.8, type=float, help='Default number of pruning bits.')
    
    @click.option('--n-bits', default=32, type=int, help='Default number of pruning bits.')
    @click.option(
        '--n-bits-act', default=32, type=int, help='Number of pruning bits for activations.'
    )
    @click.option('--per-channel', is_flag=True, help='If given, prune each channel separately.')
    @click.option('--fix-prob', is_flag=True, help='If given, fix prob.')
    
    @click.option('--mag-prune', is_flag=True, help='If given, fix prob.')
    @click.option(
        '--mag-prune-first',
        is_flag=True,
        help='MAG: prune export already done; single finetune only (skip post-train structural export).',
    )
    @click.option('--hand-ratio', default=0.15, type=float, help='manually decided pruning ratio.')
    
    @click.option('--hard', is_flag=True, help='If given, gumble softmax is hard')
    @click.option('--only-size-hard', is_flag=True, help='If given, only size loss is hard forward is soft')
    @click.option('--decay-tau', is_flag=True, help='decay temper')

    @click.option('--eta-min', default=0.01, type=float, help='Default number of pruning bits.')
    @click.option('--eta-max', default=0.5, type=float, help='Default number of pruning bits.')

    @click.option('--prune-2-4', is_flag=True, help='If given, only size loss is hard forward is soft')
    @click.option('--is-arc-prune', is_flag=True, help='If given, only size loss is hard forward is soft')
    @click.option(
        '--percentile',
        type=float,
        default=None,
        help='Percentile clipping parameter (weights and activations)',
    )
    @click.option(
        '--act-prune/--no-act-prune',
        is_flag=True,
        default=True,
        help='Run evaluation with activation pruning or use FP32 activations',
    )
    @click.option(
        '--weight-prune/--no-weight-prune',
        is_flag=True,
        default=True,
        help='Run evaluation weight pruning or use FP32 weights',
    )
    @click.option(
        '--prune-setup',
        # default='all',
        type=click.Choice(['all', 'FP_logits', 'MSE_logits']),
        default='FP_logits', # 2024-7-4 modified
        help='Method to prune the network.',
    )
    @click.option("--gating-lambda", type=float, default=0.0)
    @click.option("--single-bit", type=int, default=0)
    @click.option("--total-steps", type=int, default=0)
    @click.option("--gating-dis", type=float, default=0.0, required=True)
    @click.option(
        '--gating-method',
        type=click.Choice(["l0", "fixed", "pg"]),
        # default="l0",
        default='pg', # 2024-7-4 modified
    )
    @click.option(
        '--gate-init-dict',
        type=str,
        default=None,
    )
    @click.option("--include-pruning", is_flag=True, default=False)
    @click.option("--channel-pruning", is_flag=True, default=False)
    # @click.option("--channel-pruning10", is_flag=True, default=False)
    @click.option("--prune-only", is_flag=True, default=False)
    # @click.option("--reg-type", type=click.Choice(["base", "const", "bop", "KL", "KLweight", "L1","distil", "distilctc", "disweightout", "disweight","distilctcoutKL","distilctcattention","distilctcout","distilctcoutMSE", "distilctclayerMSE", "distilctclayerMSE248", "distilctclayerMSEstep2","distilctclayerHeadMSE","distilctcAttMSE","distilctclayerMSEnoFP","distilctcLinear","distilctclayerMSEctc4","distilctclayerMSEctc5","distilctclayerMSEnoearly","distilctclayerMSEthre","distilctclayerMSEgumble","distilctclayerMSEgumble5ctc","distilctclayerMSEgumbleKL","distilctclayerMSEgumbleKL2","distilctclayerMSEgumbleKLno2", "distilgumble111", "distilgumble5ctc111","distilgumble4ctcno4","distilgumble5ctc111mse","distilgumble5ctc111msefp8","distilgumble5ctc111mseCNN","distilgumble5ctc111mseCNNall", "distilgumble5ctc111freeze","gumbleNAS5ctc","gumbleNAS", "large", "KL","single","KLroll","baseline","KLrollNAS","KL5ctcNAS","KLrollNASq","KLrollcnnq","KLrollcnnqall","KLrollq","KLcnn","saspg","saspglora","disprune","channelpruning","channelpruning10","channelpruningwavlm","wavlm_mask","channelgate","channelgatecool","channelgatemag","channelgateL0","channelgatenorm1"]), default="const")
    @click.option("--reg-type", type=str, default="const")
    @click.option("--fixed-bit-dict", type=str, default="{'q8':0,'q4':0}")

    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        
        config.prune_cfg, remainder_kwargs = split_dict(kwargs, [
        'max_bit',
        'min_bit',
        'value_1',
        'value_075',
        'value_025',
        'value_0125',
        'value_05',
        'value_01',
        'value_0075',
        'max_prune_ratio',
        'min_prune_ratio',
        'only_size_hard',
        'decay_tau',
        'prune_2_4',
        'is_arc_prune',
        'hard',
        'fix_prob',
        'mag_prune',
        'mag_prune_first',
        'hand_ratio',
        'single_bit',
        'total_steps',
        'gating_dis',
        'gating_lambda',
        'qmethod',
        'qmethod_act',
        'weight_prune_method',
        'weight_opt_method',
        'num_candidates',
        'n_bits',
        'n_bits_act',
        'per_channel',
        'percentile',
        'act_prune',
        'weight_prune',
        'prune_setup',
        'gating_method',
        'include_pruning',
        'channel_pruning',
        # 'channel_pruning10',
        'prune_only',
        'reg_type',
        'gate_init_dict',
        'fixed_bit_dict',
        'eta_max',
        'eta_min',
        ])
        
        if kwargs['gate_init_dict'] is not None:
            gate_init_dict = eval(kwargs['gate_init_dict'])
            config.prune_cfg.gate_init_dict =  gate_init_dict
        else:
            config.prune_cfg.gate_init_dict =  {'q2':0,'q4':0, 'q8':0.02}
            
        if kwargs['fixed_bit_dict'] is not None:
            fixed_bit_dict = eval(kwargs['fixed_bit_dict'])
            config.prune_cfg.fixed_bit_dict =  fixed_bit_dict
            
            
        config.prune_cfg.qmethod_act = config.prune_cfg.qmethod_act or config.prune_cfg.qmethod

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def activation_pruning_options(func):
    @click.option(
        '--act-prune-method',
        default=RangeEstimators.running_minmax.name,
        type=click.Choice(RangeEstimators.list()),
        help='Method to determine activation pruning clipping thresholds',
    )
    @click.option(
        '--act-opt-method',
        default=OptMethod.grid.name,
        type=click.Choice(OptMethod.list()),
        help='Optimization procedure for activation pruning clipping thresholds',
    )
    @click.option(
        '--act-num-candidates',
        type=int,
        default=None,
        help='Number of grid points for grid search in MSE/Cross-entropy',
    )
    @click.option(
        '--act-momentum',
        type=float,
        default=None,
        help='Exponential averaging factor for running_minmax',
    )
    @click.option(
        '--cross-entropy-layer',
        default=None,
        type=str,
        help='Cross-entropy for activation range setting (often valuable for last layer)',
    )
    @click.option(
        '--num-est-batches',
        type=int,
        # default=1, # 2024-7-4 modified.
        default=16,
        help='Number of training batches to be used for activation range estimation',
    )
    
    @wraps(func)
    def func_wrapper(config, act_prune_method, act_opt_method, act_num_candidates, act_momentum,
                     cross_entropy_layer, num_est_batches, *args, **kwargs):
        config.act_prune = DotDict()
        config.act_prune.prune_method = act_prune_method
        config.act_prune.cross_entropy_layer = cross_entropy_layer
        config.act_prune.num_batches = num_est_batches

        config.act_prune.options = {}

        if act_num_candidates is not None: # bind with MSE
            if act_prune_method != 'MSE':
                raise ValueError('Wrong option num_candidates passed')
            else:
                config.act_prune.options['num_candidates'] = act_num_candidates

        if act_momentum is not None:
            if act_prune_method != 'running_minmax':
                raise ValueError('Wrong option momentum passed')
            else:
                config.act_prune.options['momentum'] = act_momentum

        if act_opt_method != 'grid': # default setting: act_opt_method is not used
            config.act_prune.options['opt_method'] = OptMethod[act_opt_method]
        return func(config, *args, **kwargs)

    return func_wrapper


def qat_options(func):
    # @click.option(
    #     '--clip-input',
    #     is_flag=True,
    #     default=False,
    # )
    @click.option("--checkpointing", is_flag=True, default=False)

    @click.option(
        "--learned-scale",
        type=click.Choice(["range", "scale"]),
        default="range",
        required=False,
    )
    @click.option(
        '--learn-ranges',
        is_flag=True,
        default=False,
        help='Learn pruning ranges, in that case fix ranges will be ignored.',
    )
    @click.option(
        '--fix-act-ranges/--no-fix-act-ranges',
        is_flag=True,
        default=False,
        help='Fix all activation pruning ranges for stable training',
    )
    @click.option(
        '--fix-weight-ranges/--no-fix-weight-ranges',
        is_flag=True,
        default=False,
        help='Fix all weight pruning ranges for stable training',
    )
    @wraps(func)
    def func_wrapper(config, *args, **kwargs):
        config.qat, remainder_kwargs = split_dict(
            kwargs, ['learn_ranges', 'fix_act_ranges', 'fix_weight_ranges']
        )

        return func(config, *args, **remainder_kwargs)

    return func_wrapper


def make_prune_params(config):
    weight_range_options = {}
    if config.prune_cfg.weight_prune_method in ['MSE', 'cross_entropy']: # cross_entropy inherited from MSE
        weight_range_options = dict(opt_method=OptMethod[config.prune_cfg.weight_opt_method])
    if config.prune_cfg.num_candidates is not None:
        weight_range_options['num_candidates'] = config.prune_cfg.num_candidates


    # import pdb;pdb.set_trace()
    params = {
        'lmb':config.prune_cfg.gating_lambda,
        'single_bit':config.prune_cfg.single_bit,
        'lmb_dis':config.prune_cfg.gating_dis,
        'total_steps':config.prune_cfg.total_steps,
        'weight_prune':config.prune_cfg.weight_prune,
        'act_prune':config.prune_cfg.act_prune,
        'method': PMethods[config.prune_cfg.qmethod],
        'n_bits': config.prune_cfg.n_bits,
        'n_bits_act': config.prune_cfg.n_bits_act,
        'max_bit':config.prune_cfg.max_bit,
        'min_bit':config.prune_cfg.min_bit,
        'value_1':config.prune_cfg.value_1,
        'value_0_75':config.prune_cfg.value_075,
        'value_0_5':config.prune_cfg.value_05,
        'value_0_25':config.prune_cfg.value_025,
        'value_0_125':config.prune_cfg.value_0125,
        'value_0_1':config.prune_cfg.value_01,
        'value_0_075':config.prune_cfg.value_0075,
        'max_prune_ratio':config.prune_cfg.max_prune_ratio,
        'min_prune_ratio':config.prune_cfg.min_prune_ratio,
        'fix_prob':config.prune_cfg.fix_prob, 
        'mag_prune':config.prune_cfg.mag_prune, 
        'hand_ratio':config.prune_cfg.hand_ratio, 
        'hard':config.prune_cfg.hard, 
        'only_size_hard':config.prune_cfg.only_size_hard, 
        'decay_tau':config.prune_cfg.decay_tau, 
        'prune_2_4':config.prune_cfg.prune_2_4, 
        'is_arc_prune':config.prune_cfg.is_arc_prune, 
        'per_channel_weights': config.prune_cfg.per_channel,
        'percentile': config.prune_cfg.percentile,
        "gating_method": config.prune_cfg.gating_method,
        "gate_init_dict": config.prune_cfg.gate_init_dict,
        "include_pruning": config.prune_cfg.include_pruning,
        "channel_pruning": config.prune_cfg.channel_pruning,
        # "channel_pruning10": config.prune_cfg.channel_pruning10,
        "nasp_ladder": getattr(config.prune_cfg, "nasp_ladder", False),
        "prune_only": config.prune_cfg.prune_only,
        "reg_type": config.prune_cfg.reg_type,
        "fixed_bit_dict":config.prune_cfg.fixed_bit_dict,
        'prune_setup': config.prune_cfg.prune_setup,
        'weight_range_method': RangeEstimators[config.prune_cfg.weight_prune_method],
        'weight_range_options': weight_range_options,
        'eta_max':config.prune_cfg.eta_max,
        'eta_min':config.prune_cfg.eta_min,
    }
    return params
