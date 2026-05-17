import math
import numpy as np
import matplotlib.pyplot as plt

# 定义两种衰减/增长函数
def cosine_decay(t, T, eta_max=0.5, eta_min=0.01):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))

# def cosine_growth(t, T, eta_min=0.01, eta_max=0.5):
#     return eta_min + 0.5 * (eta_max - eta_min) * (1 - np.cos(np.pi * t / T))
def cosine_growth(current_steps, total_steps, eta_min=0.01, eta_max=0.5):
    if current_steps / total_steps < 0.15:
        print(current_steps / total_steps)
        tau = (eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * current_steps / int(total_steps*0.15))))
        print(tau)
        return tau
    else:
        tau = eta_min + 0.5 * (eta_max - eta_min) * (1 - np.cos(np.pi * int(current_steps-0.15*total_steps) / int(total_steps*0.85)))
        print(tau)
    
        return tau

# 生成数据
T = 100
steps = np.arange(0, T)
decay_lr = [cosine_decay(t, T) for t in steps]
growth_lr = [cosine_growth(t, T) for t in steps]
print(growth_lr)
# 绘图
plt.figure(figsize=(12, 5))

# 衰减曲线
plt.subplot(1, 2, 1)
plt.plot(steps, decay_lr, color='blue', linewidth=2)
plt.title("Cosine Decay (0.5 → 0.01)")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True, alpha=0.3)

# 增长曲线
plt.subplot(1, 2, 2)
plt.plot(steps, growth_lr, color='red', linewidth=2)
plt.title("Cosine Growth (0.01 → 0.5)")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/project_bdda8/bdda/hnxu/prune/SSLprune/DPHuBERT_pretrain/check.png')
print('/project_bdda8/bdda/hnxu/prune/SSLprune/DPHuBERT_pretrain/check.png')