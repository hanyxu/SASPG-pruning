# SASPG SUPERB Release (48 upstream experiments)

本 release 将 DPHuBERT / SASPG 上游训练整理为 **48 条可复现实验**（`configs/experiments.csv` 中 `status=ready`）：

- 4 模型：`hubert_base`, `hubert_large`, `wav2vec2_base`, `wavlm_base`
- 2 数据规模：`100h`, `960h`
- 6 方法：`str_saspg`, `unstr_saspg`, `str_magnitude`, `unstr_magnitude`, `str_dphubert`, `unstr_dphubert`

矩阵规模：`4 × 2 × 6 = 48`。

本目录为 **upstream-only** orchestrator（无 SUPERB downstream）；统一入口为 `python3 -m core …`。

## Unified CLI (`core/`)

在 **本目录** 下执行：

| 子命令 | 用途 |
|--------|------|
| `experiments` | 对 `configs/experiments.csv` 做 `list` / `run` / `validate` |
| `pipeline` | 上游训练调度（`--run-upstream-first --exp-id …`） |
| `smoke48` | 顺序跑完全部 48 条 ready 实验（调度/连通性检查） |

旧命令见 [MIGRATION.md](MIGRATION.md)。

## 文件与脚本

- `configs/experiments.csv`：48 条实验矩阵
- `recipes/family_defaults.json`：按 family 的默认 `upstream_mode`
- `core/`：统一编排（registry / experiments / pipeline / CLI）
- `prepare_local_dependencies.sh`：从工作区 rsync `DPHuBERT*` 到 `upstream_str` / `upstream_unstr`（见 `configs/rsync_upstream_excludes.txt`，默认排除 `exp_*` 与 ckpt）
- `run_experiment.sh` → `python3 -m core experiments run`
- `run_smoke48.sh` / `run_smoke24.sh` → `python3 -m core smoke48`
- `run_smoke960_dual_gpu.sh`：单 GPU 顺序跑 48 条并写 `smoke960_logs/`（批量冒烟）

## Quick usage

首次准备（若目录内尚无 `upstream_str` / `upstream_unstr`）：

```bash
bash ./prepare_local_dependencies.sh
# 或指定源：
SASPG_SOURCE_ROOT=/abs/path/to/workspace bash ./prepare_local_dependencies.sh
```

列出 / 校验 / 单条运行：

```bash
bash ./list_experiments.sh --ready-only
bash ./validate_experiments.sh
bash ./run_experiment.sh --exp-id wav2vec2_base_960_str_saspg
python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_saspg --dry-run
```

批量 smoke48（仅调度，不替代正式训练 recipe）：

```bash
bash ./run_smoke48.sh --dry-run
bash ./run_smoke48.sh --limit 1          # 试跑 1 条
bash ./run_smoke48.sh                    # 顺序跑满 48 条
```

---

## 48 条实验：从 smoke48 到正式 setup

### smoke48 与正式训练的区别

| | smoke48 / `run_smoke48.sh` | 正式 setup（生产训练） |
|--|---------------------------|------------------------|
| 目的 | 验证 CSV、launcher 路径、环境变量、GPU 可见性 | 完整 distillation → pruning → finetune（或 magnitude 的 prune→ft） |
| 入口 | `python3 -m core smoke48` 或 `bash ./run_smoke48.sh` | 对 **单条** `exp_id` 执行 `pipeline` / `run_experiment.sh` |
| 步数 / 脚本体 | 可用 `--dry-run`；真跑时仍调用 launcher，但需你确认脚本内训练段已启用 | 必须取消 launcher 内被注释的 `distill` / `prune` / `final_distill` 等，并使用完整 `max` / `final_max` |
| 产物 | 可为空跑或短日志；**不要**把 `exp_*` / `smoke960_logs` 打进发布包 | 在 `upstream_str` 或 `upstream_unstr` 下生成 `exp_minmax*/` |

### 一次性环境准备（48 条实验共用）

在 release 根目录执行：

```bash
cd /path/to/SASPG_superb_release   # 或本 _work 目录

# 1) 上游代码树（发布包已自带则可跳过）
bash ./prepare_local_dependencies.sh

# 2) Python 环境（示例：按你集群的 conda 配置）
# source /path/to/conda.sh && conda activate dphubert

# 3) 预训练教师权重目录（四模型各一个 .hf.pth）
export DPHuBERT_PRETRAINED_DIR=/abs/path/to/pretrained
# 需包含：hubert-base-ls960.hf.pth, hubert-large-ll60k.hf.pth,
#         wav2vec2-base.hf.pth, wavlm-base-plus.hf.pth

# 4) LibriSpeech TSV（100h / 960h 由 launcher 内 train_subset 选择）
export DPHuBERT_TSV_DIR=/abs/path/to/librispeech/tsv_root
# 若未设置，scripts/source_dphubert_env.sh 会尝试
# $SASPG_ROOT/DPHuBERT_pretrain_unstr/data/librispeech

# 5) 可选：显式指定工作区根（默认自动探测）
export SASPG_ROOT=/abs/path/to/SSLprune

# 6) GPU（单卡示例）
export CUDA_VISIBLE_DEVICES=0

# 7) 校验 48 条矩阵
python3 -m core experiments validate
```

各 launcher 会通过 `scripts/source_dphubert_env.sh` 解析 `DPHuBERT_PRETRAINED_ROOT` 与 `DPHuBERT_TSV_DIR`；在 `upstream_str` / `upstream_unstr` 下执行时，`SCRIPT_DIR` 指向对应 launcher 目录。

### 各实验正式 setup（48 子块）

以下每一条对应 `configs/experiments.csv` 中的一行。正式训练时 **只跑当前 `exp_id`**，不要依赖 smoke 批量脚本的缩短配置。

#### 1. `hubert_base_100_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_100_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_base_100_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 2. `hubert_base_960_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_960_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_str_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_base_960_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 3. `hubert_large_100_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_100_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_str_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_large_100_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 4. `hubert_large_960_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_large.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_960_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_str_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_large_960_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_large.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 5. `wav2vec2_base_100_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_wav2vec2_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_100_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_str_saspg
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_100_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_wav2vec2_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 6. `wav2vec2_base_960_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_wav2vec2_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_960_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_str_saspg
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_960_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_wav2vec2_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 7. `wavlm_base_100_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_100_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_str_saspg
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_100_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 8. `wavlm_base_960_str_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh` |
| 备注 | SASPG structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_960_str_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_str_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_str_saspg
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_960_str_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 9. `hubert_base_100_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_100_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_base_100_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 10. `hubert_base_960_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_960_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_base_960_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 11. `hubert_large_100_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_100_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_large_100_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 12. `hubert_large_960_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_960_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id hubert_large_960_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 13. `wav2vec2_base_100_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_100_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_100_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 14. `wav2vec2_base_960_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_960_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_960_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 15. `wavlm_base_100_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_100_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_100_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 16. `wavlm_base_960_unstr_saspg`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_saspg`（`saspg` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh` |
| 备注 | SASPG unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_960_unstr_saspg`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_unstr_saspg --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_unstr_saspg
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_960_unstr_saspg
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 17. `hubert_base_100_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_hubert_base_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_100_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_base_100_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_hubert_base_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 18. `hubert_base_960_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_hubert_base_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_960_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_base_960_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_hubert_base_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 19. `hubert_large_100_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_100_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_large_100_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 20. `hubert_large_960_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_960_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_large_960_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 21. `wav2vec2_base_100_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_wav2vec2_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_100_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_100_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_wav2vec2_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 22. `wav2vec2_base_960_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_wav2vec2_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_960_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_960_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_wav2vec2_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 23. `wavlm_base_100_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_100_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_100_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 24. `wavlm_base_960_str_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm_mag.sh` |
| 备注 | magnitude structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_960_str_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_str_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_str_magnitude
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_960_str_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm_mag.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 25. `hubert_base_100_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_hubert_base_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_100_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_base_100_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_hubert_base_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 26. `hubert_base_960_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_hubert_base_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_960_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_base_960_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_hubert_base_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 27. `hubert_large_100_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7002_unstr_buffer_large_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_100_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_large_100_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7002_unstr_buffer_large_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 28. `hubert_large_960_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7002_unstr_buffer_large_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_960_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id hubert_large_960_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7002_unstr_buffer_large_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 29. `wav2vec2_base_100_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_100_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_100_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 30. `wav2vec2_base_960_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_960_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_960_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 31. `wavlm_base_100_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_100_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_100_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 32. `wavlm_base_960_unstr_magnitude`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_magnitude`（`magnitude` / `prune_then_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm_mag.sh` |
| 备注 | magnitude unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_960_unstr_magnitude`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_unstr_magnitude --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_unstr_magnitude
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_960_unstr_magnitude
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm_mag.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 33. `hubert_base_100_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_2e-4_0.5_0.01_warm0.3.sh` |
| 备注 | DPHuBERT gate_lr 2e-4 recipe |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_100_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_base_100_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_2e-4_0.5_0.01_warm0.3.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 34. `hubert_base_960_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_2e-4_0.5_0.01_warm0.3.sh` |
| 备注 | DPHuBERT gate_lr 2e-4 recipe |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_960_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_base_960_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_2e-4_0.5_0.01_warm0.3.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 35. `hubert_large_100_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large.sh` |
| 备注 | DPHuBERT large structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_100_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_large_100_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 36. `hubert_large_960_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large.sh` |
| 备注 | DPHuBERT large structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_960_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_large_960_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_large.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 37. `wav2vec2_base_100_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_wav2vec2_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_conv.sh` |
| 备注 | DPHuBERT conv variant |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_100_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_100_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_wav2vec2_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_conv.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 38. `wav2vec2_base_960_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_wav2vec2_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_conv.sh` |
| 备注 | DPHuBERT conv variant |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_960_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_960_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_wav2vec2_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_conv.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 39. `wavlm_base_100_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh` |
| 备注 | DPHuBERT wavlm structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_100_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_100_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 40. `wavlm_base_960_str_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `str_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_str/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh` |
| 备注 | DPHuBERT wavlm structured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_960_str_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_str_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_str_dphubert
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_960_str_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_str && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_wavlm.sh
   ```
7. 产物写入 `upstream_str/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 41. `hubert_base_100_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_unstr_buffer.sh` |
| 备注 | DPHuBERT unstr buffer tau 0.75 |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_100_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_100_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_base_100_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_unstr_buffer.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 42. `hubert_base_960_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_unstr_buffer.sh` |
| 备注 | DPHuBERT unstr buffer tau 0.75 |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-base-ls960.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_base_960_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_base_960_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_base_960_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.75_unstr_buffer.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 43. `hubert_large_100_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh` |
| 备注 | DPHuBERT large unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_100_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_100_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_large_100_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 44. `hubert_large_960_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `hubert_large` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh` |
| 备注 | DPHuBERT large unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/hubert-large-ll60k.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id hubert_large_960_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id hubert_large_960_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id hubert_large_960_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.70_unstr_buffer_large.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 45. `wav2vec2_base_100_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh` |
| 备注 | DPHuBERT wav2vec2 unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_100_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_100_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_100_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 46. `wav2vec2_base_960_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wav2vec2_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh` |
| 备注 | DPHuBERT wav2vec2 unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wav2vec2-base.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wav2vec2_base_960_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wav2vec2_base_960_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id wav2vec2_base_960_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wav2vec2.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 47. `wavlm_base_100_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `100h`（`train_subset=train100`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh` |
| 备注 | DPHuBERT wavlm unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_100_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_100_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_100_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_100h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


#### 48. `wavlm_base_960_unstr_dphubert`

| 项 | 值 |
|----|-----|
| 模型 | `wavlm_base` |
| 数据 | `960h`（`train_subset=train960`） |
| 方法 | `unstr_dphubert`（`dphubert` / `distill_prune_ft`） |
| Launcher | `upstream_unstr/run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh` |
| 备注 | DPHuBERT wavlm unstructured |

**smoke48 → 正式训练：** 批量冒烟用 `bash ./run_smoke48.sh`（或 `python3 -m core smoke48`）只验证调度与脚本可解析；正式实验需对本条单独执行完整上游 recipe（见下），并确认 launcher 内 `distill` / `prune` / `final_distill` 等 Python 调用已启用、步数为论文/生产配置。

1. `cd` 到本 release 根目录，完成上文「一次性环境准备」。
2. 确认教师 checkpoint：`$DPHuBERT_PRETRAINED_ROOT/wavlm-base-plus.hf.pth`。
3. （可选）`python3 -m core experiments validate`；`python3 -m core experiments list --exp-id wavlm_base_960_unstr_dphubert`。
4. （可选）干跑：`python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_unstr_dphubert --dry-run`。
5. **正式运行**（在已设置 `CUDA_VISIBLE_DEVICES` 的节点上）：
   ```bash
   python3 -m core pipeline --run-upstream-first --exp-id wavlm_base_960_unstr_dphubert
   # 或：bash ./run_experiment.sh --exp-id wavlm_base_960_unstr_dphubert
   ```
6. 等价于在 launcher 目录执行：
   ```bash
   cd upstream_unstr && bash -euo pipefail run_960h_test_thre_lr_2e-4_320x2_gate_down_1e-4_0.5_0.01_warm0.3_0.7503_unstr_buffer_wavlm.sh
   ```
7. 产物写入 `upstream_unstr/exp_minmax*/`（由脚本内 `root_dir` 命名）；勿将本地 `exp_*` 打进发布包。


