# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging

from utils.utils import pass_data_for_range_estimation,pass_data_for_range_estimation_width


# setup logger
logger = logging.getLogger('QAT')
logger.setLevel('INFO')


def prepare_model_for_pruning(config, model, loader):

    # import pdb;pdb.set_trace()
    # estimate ranges using training data
    pass_data_for_range_estimation(
        loader=loader,
        model=model,
        act_prune=config.prune_cfg.act_prune,
        weight_prune=config.prune_cfg.weight_prune,
        max_num_batches=config.act_prune.num_batches,
        cross_entropy_layer=config.act_prune.cross_entropy_layer,
    )
    # pruner的前馈是一定会做的，但Min max不一定会改变（是esimate range决定的）
    # import pdb;pdb.set_trace()
    # put pruners in desirable state
    if config.qat.learn_ranges:
        logger.info('Make pruners learnable')
        model.learn_ranges() # 训练的过程中是fix的
        # 先让 delta 和 zero 变为 nn.parameters 从而可以QAT
    else:
        logger.info(
            f'Fix pruner ranges to fixW={config.qat.fix_weight_ranges} and '
            f'fixA={config.qat.fix_act_ranges}'
        )
    
    # class Qstates(Enum):
    # estimate_ranges = 0  # ranges are updated in eval and train mode
    # fix_ranges = 1  # pruning ranges are fixed for train and eval
    # learn_ranges = 2  # pruning params are nn.Parameters
    # estimate_ranges_train = 3  # pruning ranges are updated during train and fixed for eval
        logger.info('Make pruners estimate_ranges_train:')
        # freeze pruning ranges if applicable
        model.estimate_ranges_train()  # we use updating ranges in training as default # One 
        # 对min max的设置 会再训练的时候更新
        # 前面前馈的x 先得到一个大致的
        if config.qat.fix_weight_ranges:
            model.fix_weight_ranges()
        if config.qat.fix_act_ranges:
            model.fix_act_ranges()

    # ensure we have the desired prune state
    model.set_prune_state(config.prune_cfg.weight_prune, config.prune_cfg.act_prune)
    return model


def get_qweight_num_hubert(model, num_layer=None, component=None):
    
    qweight_num = [0] * num_layer
    
    if component:
        for j in range(num_layer):
            module = list(model.hubert.encoder.layers[j].named_children())[j][1]
            for name, layer in module.named_children():
                    if name == component:
                        qweight_num[j] += layer.qweight_num()
        
        return qweight_num
    else:
        for component in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'feed_forward.1','feed_forward.2']:
            for j in range(num_layer):
                module = list(model.hubert.encoder.layers[j].named_children())[j][1]
                for name, layer in module.named_children():
                        if name == component:
                            qweight_num[j] += layer.qweight_num()
        
        return qweight_num

def get_qweight_num_wav2vec2(model, num_layer=None, component=None):
    
    qweight_num = [0] * num_layer
    # fp_model, _ = load_trained_model(fp_path)
    # load_model here
    
    if component:
        for j in range(num_layer):
            module = list(model.wav2vec2.encoder.layers[j].named_children())[j][1]
            for name, layer in module.named_children():
                    if name == component:
                        qweight_num[j] += layer.qweight_num()
        
        return qweight_num
    else:
        for component in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'feed_forward.1','feed_forward.2']:
            for j in range(num_layer):
                module = list(model.wav2vec2.encoder.layers[j].named_children())[j][1]
                for name, layer in module.named_children():
                        if name == component:
                            qweight_num[j] += layer.qweight_num()
        
        return qweight_num


 
def prepare_model_for_pruning_width(config, model, loader):

    # import pdb;pdb.set_trace()
    # estimate ranges using training data
    pass_data_for_range_estimation_width(
        loader=loader,
        model=model,
        act_prune=config.prune_cfg.act_prune,
        weight_prune=config.prune_cfg.weight_prune,
        max_num_batches=config.act_prune.num_batches,
        cross_entropy_layer=config.act_prune.cross_entropy_layer,
    )
    # pruner的前馈是一定会做的，但Min max不一定会改变（是esimate range决定的）
    # import pdb;pdb.set_trace()
    # put pruners in desirable state
    if config.qat.learn_ranges:
        logger.info('Make pruners learnable')
        model.learn_ranges() # 训练的过程中是fix的
        # 先让 delta 和 zero 变为 nn.parameters 从而可以QAT
    else:
        logger.info(
            f'Fix pruner ranges to fixW={config.qat.fix_weight_ranges} and '
            f'fixA={config.qat.fix_act_ranges}'
        )
    
    # class Qstates(Enum):
    # estimate_ranges = 0  # ranges are updated in eval and train mode
    # fix_ranges = 1  # pruning ranges are fixed for train and eval
    # learn_ranges = 2  # pruning params are nn.Parameters
    # estimate_ranges_train = 3  # pruning ranges are updated during train and fixed for eval
        logger.info('Make pruners estimate_ranges_train:')
        # freeze pruning ranges if applicable
        model.estimate_ranges_train()  # we use updating ranges in training as default # One 
        # 对min max的设置 会再训练的时候更新
        # 前面前馈的x 先得到一个大致的
        if config.qat.fix_weight_ranges:
            model.fix_weight_ranges()
        if config.qat.fix_act_ranges:
            model.fix_act_ranges()

    # ensure we have the desired prune state
    model.set_prune_state(config.prune_cfg.weight_prune, config.prune_cfg.act_prune)
    return model
