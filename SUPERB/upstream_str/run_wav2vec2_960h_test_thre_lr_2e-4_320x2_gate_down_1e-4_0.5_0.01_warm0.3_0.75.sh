#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240000M
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=dphubert
#SBATCH --time=2-00:00:00

# first source conda.sh, and then
# activate your conda environment

set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../scripts/source_dphubert_env.sh"


# shared config
tsv_dir=${DPHuBERT_TSV_DIR:-data/librispeech}        # data path
train_subset=train960           # train subset name: train960, train100
# teacher_ckpt=${DPHuBERT_PRETRAINED_ROOT}/hubert-base-ls960.hf.pth    # checkpoint path
teacher_ckpt=${SASPG_ROOT}/DPHuBERT_pretrain_unstr/pretrained/wav2vec2-base.hf.pth    # checkpoint path
student_ckpt=${teacher_ckpt}    # student initialization, same as teacher
distill_layers=0.4,8,12         # use period to separate groups where each group shares the same linear layer: [0], [4, 8, 12]
distill_mode=layer2layer        # "layer2layer", "predlayer"
l2_weight=0             # weight for L2 loss
l1_weight=1             # weight for L1 loss
cos_weight=1            # weight for cosine similarity
cos_type=raw            # "raw", "log_sig"

# distill config
lr=0.0002               # learning rate
warmup=15000            # warmup steps
max=50000               # max update steps
pruning_units=head,interm      # conv,head,interm,attlayer,ffnlayer
reg_lr=0.02             # learning rate for regularization params
gate_lr=1e-4             
target_sparsity=0.7495    # final target sparsity
sparsity_warmup=5000    # warmup steps for sparsity; sparsity will linearly increase from 0 to target
seconds_per_batch=320
accum_grad=2
eta_max=0.5
eta_min=0.01

root_dir=exp_minmax_960h_wav2vec2/wav2vec2-base_${train_subset}_sp${target_sparsity}_spup${sparsity_warmup}_lr${lr}_up${warmup}_max${max}_${distill_mode}${distill_layers}_reglr${reg_lr}_gatelr${gate_lr}_${pruning_units}_sec${seconds_per_batch}_accum${accum_grad}_gate_norm_${eta_max}_${eta_min}_tau_gate_up_30%_down_try_try_try

# 之前是10 tau 0.5-> 0.01 50000tausteps
# final distill config
final_lr=0.0001         # learning rate for final distillation (training step 2)
final_warmup=5000       # warmup steps
final_max=25000         # max update steps
final_exp_dir=${root_dir}/lr${final_lr}_up${final_warmup}_max${final_max}


# Training step 1: distill
mkdir -p ${root_dir}

# 持续检测2号显卡显存
# while :; do
#     # 获取2号显卡的可用显存（MiB）并去除空格
#     mem=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | xargs)
    
#     # 检查显存是否大于20000MiB
#     if [ "$mem" -ge 42000 ]; then
#         echo "GPU 2 has sufficient memory (${mem}MiB). Proceeding..."
#         break
#     else
#         echo "Waiting for GPU 2 memory (>42000MiB). Current: ${mem}MiB"
#         sleep 600  # 等待60秒后重试
#     fi
# done

python distill_1_01.py \
    --tsv_dir ${tsv_dir} \
    --train_subset ${train_subset} \
    --seconds_per_batch ${seconds_per_batch} \
    --num_workers 12 \
    --exp_dir ${root_dir} \
    --log_interval 50 \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_updates ${warmup} \
    --max_updates ${max} \
    --eta_min ${eta_min} \
    --eta_max ${eta_max} \
    --clip_norm 10.0 \
    --num_nodes 1 \
    --gpus 1 \
    --accum_grad ${accum_grad} \
    --precision 16 \
    --teacher_ckpt ${teacher_ckpt} \
    --student_ckpt ${student_ckpt} \
    --distill_layers ${distill_layers} \
    --distill_mode ${distill_mode} \
    --l2_weight ${l2_weight} \
    --l1_weight ${l1_weight} \
    --cos_weight ${cos_weight} \
    --cos_type ${cos_type} \
    --pruning_units ${pruning_units} \
    --reg_learning_rate ${reg_lr} \
    --gate_learning_rate ${gate_lr} \
    --target_sparsity ${target_sparsity} \
    --sparsity_warmup_updates ${sparsity_warmup} 2>&1 | tee ${root_dir}/distill.log || exit 1;

# prune and save model
python prune_1_01_wav2vec2.py \
    --eta_min ${eta_min} \
    --eta_max ${eta_max} \
    --distilled_ckpt ${root_dir}/ckpts/*.ckpt \
    --original_ckpt ${student_ckpt} || exit 1;


# Training step 2: final distill
pruned_ckpt=${root_dir}/ckpts/pruned_wav2vec2_base.pth
mkdir -p ${final_exp_dir}

python final_distill.py \
    --tsv_dir ${tsv_dir} \
    --train_subset ${train_subset} \
    --seconds_per_batch ${seconds_per_batch} \
    --num_workers 12 \
    --exp_dir ${final_exp_dir} \
    --log_interval 50 \
    --learning_rate ${final_lr} \
    --weight_decay 0.0 \
    --warmup_updates ${final_warmup} \
    --max_updates ${final_max} \
    --clip_norm 10.0 \
    --num_nodes 1 \
    --gpus 1 \
    --accum_grad ${accum_grad} \
    --precision 16 \
    --teacher_ckpt ${teacher_ckpt} \
    --student_ckpt ${pruned_ckpt} \
    --distill_layers ${distill_layers} \
    --distill_mode ${distill_mode} \
    --l2_weight ${l2_weight} \
    --l1_weight ${l1_weight} \
    --cos_weight ${cos_weight} \
    --cos_type ${cos_type} 2>&1 | tee ${final_exp_dir}/final_distill_prune_true_1_01.log || exit 1;

# save final model and config
python save_final_ckpt_wav2vec2.py \
    --config_path ${pruned_ckpt} \
    --ckpt_after_final_distill ${final_exp_dir}/ckpts/*.ckpt || exit 1;
