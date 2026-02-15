  
> `Q-learning_CartPole.py`（表格型 Q-learning）、`山地车.py`（DQN）、`控制一个摆.py`（DDPG）。

---
## 1 你不需要录视频：RL 的“结果”怎么量化

  

强化学习的训练过程是随机的，最靠谱的“成果展示”通常是 **曲线 + 指标**，而不是视频。

  

推荐你在 Obsidian 里固定记录这几项（每个任务都通用）：

  

1. **训练曲线（learning curve）**

   - x：episode

   - y：每个 episode 的 return（累计奖励）

   - 再叠一条 **滑动平均**（比如 50 或 100）

2. **评估指标（Evaluation）**

   - 用 *固定策略*（一般是 greedy / no-noise / epsilon=0）跑 N 个 episode

   - 记录：`eval_avg_return`、`success_rate`、（可选）`avg_steps_success`

3. **复现信息（Reproducibility）**

   - seed、episodes、关键超参、版本（gymnasium/torch）

  

> 只有当你需要直观展示“学会了什么动作”时，才补一个 GIF（5 秒即可）；但绝大多数学习笔记里，用曲线和数字就够了。

  

---

  

## 2 关键概念（够用版）

  

- **MDP**：状态 `s`、动作 `a`、奖励 `r`、转移 `P(s'|s,a)`、折扣 `γ`

- **Return**：`G = Σ γ^t r_t`（通常你画的就是每个 episode 的 return）

- **探索/利用**：

  - 离散动作常用 epsilon-greedy（ε）

  - 连续动作常用加噪声（如 OU noise）

- **On-policy vs Off-policy**：

  - Q-learning / DQN / DDPG 都是 off-policy（能用 replay buffer 提升样本效率）

  

---

  

## 3 CartPole（表格型 Q-learning）——`Q-learning_CartPole.py`

  

### 3.1 任务直觉

  

CartPole 的目标是让杆子不倒：每一步都得到 `+1`，所以 **return≈撑住的步数**（越高越好）。

  

### 3.2 为什么要离散化

  

CartPole 的状态是连续的 4 维向量（位置/速度/角度/角速度）。表格型 Q-learning 需要一个有限大小的 Q 表：

  

- 先把连续空间切成 `bins` 个桶：`s_cont -> s_disc`

- 再对离散状态做更新：`Q[s,a]`

  

### 3.3 核心更新式

  

`Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]`

  

### 3.4 你应该量化什么

  

- `avg_reward(window=100)`：训练曲线是否稳定上升

- `solved_episode`：满足阈值的最早 episode（例如 avg100 ≥ 475）

- `eval_avg_reward`：评估时（greedy）平均 return

  

脚本会输出：

  

- `cartpole_qlearning_training.png`

- `cartpole_qlearning_summary.json`

- `cartpole_qlearning_report.md`

- `cartpole_q_table.npy`

  

---

  

## 4 MountainCar（DQN）——`山地车.py`

  

### 4.1 任务直觉

  

MountainCar 的奖励很“硬”：每一步 -1，直到到达山顶才结束。  

所以 raw return 一般是 `-steps`，**越接近 0 越好**；更直观的是看 **成功率** 和 **成功所需步数**。

  

### 4.2 DQN 解决了什么问题

  

表格法在连续状态上会爆炸，所以 DQN 用神经网络近似 Q 函数：`Q(s,a)≈Q_θ(s,a)`。

  

常见稳定化技巧：

  

- **Replay Buffer**：打破时间相关性

- **Target Network**：降低 bootstrap 的抖动（每隔 N episode/step 同步一次）

- **Epsilon decay**：先探索后利用

  

脚本训练时对奖励做了轻量 shaping（加速度项），但仍然会记录 **raw reward**（更利于跨实验对比）。

  

### 4.3 你应该量化什么

  

- `success_rate(rolling 50)`：最近 50 个 episode 成功比例

- `avg_steps_success`：成功时平均步数（越低越好）

- `eval_success_rate`：评估集成功率（greedy）

  

脚本会输出：

  

- `mountaincar_dqn_training_results.png`

- `mountaincar_dqn_summary.json`

- `mountaincar_dqn_report.md`

- `mountaincar_dqn.pth`

  

---

  

## 5 Pendulum（DDPG）——`控制一个摆.py`

  

### 5.1 任务直觉

  

Pendulum 的动作是连续的（扭矩），需要把摆保持在竖直附近。奖励通常为负值，越接近 0 越好（不同实现尺度略有差别）。

  

### 5.2 DDPG 的关键点（够用版）

  

- **Actor**：输出连续动作 `a = π(s)`

- **Critic**：估计 Q 值 `Q(s,a)`

- **OU noise**：连续动作探索

- **软更新（Polyak）**：`θ_target ← τ θ + (1−τ) θ_target`

  

### 5.3 你应该量化什么

  

- `avg_return(last20)`：训练末尾的平均回报（Pendulum 波动较大，看滑动平均更靠谱）

- `eval_avg_return(no-noise)`：评估集平均回报（禁用噪声）

  

脚本会输出：

  

- `pendulum_ddpg_training.png`

- `pendulum_ddpg_summary.json`

- `pendulum_ddpg_report.md`

- `pendulum_ddpg_actor.pth` / `pendulum_ddpg_critic.pth`

  

---

  

## 6 复现命令（建议先 quick 再 full）

  

```powershell

# 1) CartPole（表格 Q-learning）

python .\Q-learning_CartPole.py

  

# 2) MountainCar（DQN）

python .\山地车.py --episodes 300

  

# 3) Pendulum（DDPG）

python .\控制一个摆.py --episodes 300

```

  

> 训练时间随机器和是否 GPU 有差异；先用 200~300 episodes 看曲线能否上升，再决定是否拉满到默认 500。

  

---

  

## 7 本次运行结果（2026-02-15 ~ 2026-02-16）

  

> 说明：这三项训练是在 2026-02-15 晚上到 2026-02-16 凌晨跑出来的，所以时间戳跨了 0 点。  

> 结果以各脚本生成的 `*_summary.json` / `*_report.md` 为准；图片（`*.png`）直接放在 Obsidian 同目录即可显示。

  

### 7.1 CartPole（表格 Q-learning）

  

- 时间：`2026-02-15 23:57:43`

- solved：avg100 ≥ 475（本次未达到，`solved_episode=None`）

- 训练末尾 avg100：`215.0`

- Eval（20 episodes, greedy）avg reward：`106.0`

- 产物：`cartpole_qlearning_training.png` `cartpole_qlearning_report.md` `cartpole_qlearning_summary.json`

  

> 注：表格法对离散化非常敏感；想让它更稳定地“解出来”，通常需要继续调 bins/边界/探索日程，或直接换成 DQN 这类函数逼近方法。

  

### 7.2 MountainCar（DQN）

  

- 时间：`2026-02-15 23:58:02`

- 训练：success=`31/300`（`10.3%`）| avg_reward_last50=`-181.1`

- Eval（20 episodes, greedy）：success_rate=`100.0%` | avg_steps_success=`153.55`

- 产物：`mountaincar_dqn_training_results.png` `mountaincar_dqn_report.md` `mountaincar_dqn_summary.json` `mountaincar_dqn.pth`

  

### 7.3 Pendulum（DDPG）

  

- 时间：`2026-02-16 00:01:35`

- 训练末尾 avg20 return：`-156.4`

- Eval（10 episodes, no-noise）avg return：`-86.9`

- 产物：`pendulum_ddpg_training.png` `pendulum_ddpg_report.md` `pendulum_ddpg_summary.json` `pendulum_ddpg_actor.pth` `pendulum_ddpg_critic.pth`