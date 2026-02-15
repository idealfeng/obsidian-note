  
> `Q-learning_CartPole.py`（表格型 Q-learning）、`山地车.py`（DQN）、`控制一个摆.py`（DDPG）。

---
## 1 核心概念速览

**强化学习 = 智能体通过试错学习最优策略**
```
环境(Environment)
    ↓ 状态(State)
智能体(Agent)
    ↓ 动作(Action)
环境(Environment)
    ↓ 奖励(Reward) + 新状态
智能体更新策略
    ↓ 循环...
```

**目标：** 最大化累计奖励（不只看眼前，还要考虑长远）

---

## 2 三种算法对比

| 算法             | 环境          | 状态空间    | 动作空间   | 核心思想           |
| -------------- | ----------- | ------- | ------ | -------------- |
| **Q-Learning** | CartPole    | 离散(表格)  | 离散(2个) | 查表学习Q值         |
| **DQN**        | MountainCar | 连续→神经网络 | 离散(3个) | 神经网络拟合Q函数      |
| **DDPG**       | Pendulum    | 连续      | 连续     | Actor-Critic架构 |

**递进关系：** 表格法 → 值函数逼近 → 策略梯度

---

## 3 Q-Learning：最简单的强化学习

### 3.1 任务：CartPole (平衡杆子)

**目标：** 通过左右移动小车，让杆子保持直立
- 状态：位置、速度、角度、角速度（4维）
- 动作：左移(0) 或 右移(1)
- 奖励：每存活1步 = +1分

---

### 3.2 奖励函数： **CartPole 最简单的生存奖励**

**原始奖励函数：**
```python
reward = +1  # 每存活一步
done = True if (杆子角度>12° 或 小车出界)
```

特点：
- ✅ 密集奖励：每步都有反馈
- ✅ 目标明确：活得越久越好
- ✅ 无需塑形：奖励天然对齐目标

为什么有效？
- 任务简单，二元动作（左/右）
- 倒下 = episode结束 = 隐式惩罚
- γ=0.99让智能体关注长期生存

---
### 3.3 核心原理：Q表（查表法）

**Q(s, a) = 在状态s采取动作a的"价值"**
```
      动作0(左)  动作1(右)
状态1   0.5       0.8     ← 选右边（更高）
状态2   0.3       0.2     ← 选左边
状态3   ...       ...
```

**更新公式（Bellman方程）：**
$$
Q(s,a)\leftarrow Q(s,a)
+ \alpha \Big(
\underbrace{r+\gamma\max_{a^{\prime}}Q(s^{\prime},a^{\prime})}_{\mathrm{target}}
-
\underbrace{Q(s,a)}_{\mathrm{current}}
\Big)
$$

- **α (学习率)**：新旧信息的权重，通常0.1-0.5
- **γ (折扣因子)**：未来奖励的重要性，通常0.9-0.99

---

### 3.4 关键技术：状态离散化

**问题：** CartPole状态是连续的（如位置-4.8到4.8）

**解决：** 把连续值切成有限个"桶"
```python
# 示例：位置(-4.8, 4.8)切成6份
bins = 6
edges = [-4.8, -2.88, -0.96, 0.96, 2.88, 4.8]

# 位置1.5 → 落在桶3
# 位置-3.0 → 落在桶1
```

**最终Q表大小：** 6 × 12 × 6 × 12 × 2 = 10368个值

位置桶数 × 速度桶数 × 角度桶数 × 角速度桶数 × 动作数

---

### 3.5 ε-贪心策略（Exploration vs Exploitation）

**探索-利用困境：**
- 总选最优动作 → 可能错过更好策略（利用）
- 随机探索 → 学习慢（探索）

**解决：ε-greedy**
```python
if random() < ε:
    action = 随机动作  # 探索
else:
    action = argmax Q(s, a)  # 利用
```

**ε衰减：** 1.0 → 0.01（从大量探索到几乎不探索）

---

### 3.6 训练流程
```
初始化Q表(全0)
for 每个episode:
    重置环境 → 获得初始状态s
    for 每一步:
        1. 选择动作 a (ε-greedy)
        2. 执行 a → 得到 reward, 新状态s'
        3. 更新 Q(s,a) ← α × TD_target
        4. s ← s'
        5. 如果杆子倒了/时间到 → 结束episode
```

**判定成功：** 最近100轮平均奖励 ≥ 475（满分500）

### 3.7 结果

<div style="display: flex; gap: 10px; justify-content: center;">
  <img src="_attachments/cartpole_first.gif" style="width:300px; height:300px; object-fit: cover; object-position: bottom;">
  <img src="_attachments/cartpole_last.gif" style="width:300px; height:300px; object-fit: cover;">
</div>

---

## 4 DQN：用神经网络代替Q表

### 4.1 任务：MountainCar (登山小车)

**目标：** 小车冲上右侧山顶（引擎动力不够，需借助左侧助跑）
- 状态：位置、速度（2维，连续）
- 动作：左推(0)、不动(1)、右推(2)
- 奖励：每步-1（鼓励快速完成），到达山顶终止

---
### 4.2 奖励函数：MountainCar 稀疏奖励的地狱

**原始奖励函数：**
```python
reward = -1  # 每一步
done = True if (到达山顶 或 超时200步)
```

**问题分析：**
- ❌ **极度稀疏**：99%的episode没到过山顶 → 全是-200
- ❌ **无方向性**：智能体不知道怎么改进
- ❌ **局部最优**：可能学会"躺平"（最小化步数 = 直接超时）

**Reward Shaping（奖励塑形）：**
```python
# 原始
reward = -1

# 塑形后
position, velocity = next_state
shaped_reward = -1 + abs(velocity) * 5.0
#                    └─────┬─────┘
#                    鼓励速度大（冲坡需要加速）
```

**改进效果：**
```
原始奖励：-200, -200, -200, -200, -150（偶然到达）
塑形奖励：-120, -115, -100, -85, -60（逐步改进）
         ↑ 智能体能看到"进步"
```

**⚠️ 塑形风险：**
- 过度强调速度 → 可能学会"来回摇摆"而不是登顶
- 奖励设计偏差 → 学到错误策略

---
### 4.3 为什么需要DQN？

**Q-Learning的局限：**
- 状态空间太大 → Q表爆炸（围棋有10^170种状态）
- 连续状态难以离散化 → 信息损失

**DQN的解决：** 用神经网络拟合Q函数
```
状态(s) → 神经网络 → Q值 [Q(s,a0), Q(s,a1), Q(s,a2)]
                            ↑
                    一次输出所有动作的Q值
```

---

### 4.4 核心创新：经验回放 (Experience Replay)

**问题：** 连续样本高度相关 → 网络容易过拟合

**解决：** 把经验存到记忆池，随机抽样训练
```python
# 每一步存入记忆
memory.push(state, action, reward, next_state, done)

# 训练时随机抽batch
batch = memory.sample(batch_size=64)
→ 打破样本相关性，提高数据利用率
```

---

### 4.5 核心创新：目标网络 (Target Network)

**问题：** 用同一网络计算当前Q和目标Q → 不稳定（追逐移动目标）

**解决：** 维护两个网络
```
Policy Net (主网络)      Target Net (目标网络)
    ↓ 每步更新                ↓ 每N步同步
Q(s,a) 当前值           max Q(s',a') 目标值
```

**更新规则：**
```python
# Loss函数
TD_target = reward + γ × Target_Net(s').max()
Loss = MSE(Policy_Net(s,a), TD_target)

# 每10轮同步一次
if episode % 10 == 0:
    Target_Net ← Policy_Net
```

---

### 4.6 Reward Shaping（奖励塑形）

**问题：** 原始奖励稀疏（只有到达山顶才有正反馈）

**技巧：** 加入中间奖励引导学习
```python
# 原始奖励：每步-1
# 塑形后：reward + abs(velocity) × 5
# → 鼓励小车加速（即使还没到山顶）
```

⚠️ **注意：** 过度塑形可能导致局部最优

---

### 4.7 网络结构
```
Input: 状态 [position, velocity]
    ↓
Linear(2 → 128) + ReLU
    ↓
Linear(128 → 128) + ReLU
    ↓
Linear(128 → 3)  # 3个动作的Q值
```

**简单就够用** - MountainCar状态简单，不需要深层网络

### 4.8 结果

<div style="display: flex; gap: 10px; justify-content: center;">
  <img src="_attachments/mountaincar_first.gif" style="width:300px; height:300px; object-fit: cover; object-position: bottom;">
  <img src="_attachments/mountaincar_last.gif" style="width:300px; height:300px; object-fit: cover;">
</div>

---

## 5 DDPG：连续动作空间的挑战

### 5.1 任务：Pendulum (倒立摆)

**目标：** 控制扭矩让摆竖直向上
- 状态：cos(θ), sin(θ), 角速度（3维）
- 动作：扭矩 **[-2.0, 2.0]** （连续！）
- 奖励：越接近竖直向上越高（-16到0）

---

### 5.2 奖励函数：Pendulum 连续负奖励

**原始奖励函数：**
```python
theta = arctan2(sin_theta, cos_theta)  # 角度
reward = -(theta² + 0.1*theta_dot² + 0.001*action²)
#         └──┬──┘   └─────┬─────┘   └─────┬─────┘
#          角度偏离    角速度过大      动作能耗
```

设计哲学：
- 多目标平衡：
  - 主要目标：竖直向上（theta=0）
  - 次要目标：稳定（角速度小）
  - 能耗约束：动作不要太剧烈

奖励范围： -16.27（最差）到 0（完美）

特点：
- ✅ 密集+连续：每步都有细粒度反馈
- ✅ 可微分：适合梯度优化
- ✅ 物理直觉：像控制成本函数

**为什么不用稀疏奖励？**
```python
# 如果用这个：
reward = 1 if abs(theta) < 0.1 else 0
# → 连续控制任务很难碰巧进入目标区域
# → 学习信号太少
```

---

### 5.3 为什么DQN不适用？

**问题：** DQN只能处理离散动作
```
# DQN输出：[Q(左), Q(中), Q(右)]
# Pendulum需要：精确控制力矩 如 1.37N·m

# 暴力离散化？ 
# 100个力矩档位 → 动作空间爆炸
# 精度不够 → 控制效果差
```

---

### 5.4 DDPG核心思想：Actor-Critic架构

**两个神经网络分工：**
```
        Actor（演员）              Critic（评论家）
           ↓                          ↓
    策略网络 π(s)               价值网络 Q(s,a)
    输入状态 → 输出动作         输入(状态,动作) → 输出Q值
    "我要做什么"                "你做得怎么样"
```

**训练流程：**
1. Actor根据状态s选择动作a
2. Critic评估这个(s,a)的价值Q(s,a)
3. Actor根据Critic的反馈调整策略（梯度上升）
4. Critic根据真实奖励更新Q值（TD learning）

---

### 5.5 关键技术1：OU噪声探索

**问题：** 连续动作空间怎么探索？

**解决：** Ornstein-Uhlenbeck噪声（有记忆的随机游走）
```python
# 不是简单加随机数
noise = OUNoise()  # 平滑的探索噪声
action = actor(state) + noise.sample()

# 特点：时间相关性 → 动作更平滑
```

**为什么不用高斯噪声？** 物理系统需要平滑控制，突变噪声会导致震荡

---

### 5.6 关键技术2：软更新 (Soft Update)

DQN的硬更新：Target 网络参数一段时间不动；到第 N 步时，把 Policy 的参数整份复制过去。
```python
# 每N步完全复制
if step % N == 0:
    Target ← Policy  # 突变
```

DDPG的软更新：每一步都更新一点点，让 Target 朝 Policy 靠近
```python
# 每步小幅度混合
Target ← τ × Policy + (1-τ) × Target
# τ = 0.005 → 每步只更新0.5%
```

**优势：** 更稳定，减少震荡

---

### 5.7 网络结构

**Actor（状态 → 动作）：**
```
Input: 状态 [cos(θ), sin(θ), ω]
    ↓
Linear(3 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 1) + Tanh  # 输出[-1,1]
    ↓
× max_action  # 缩放到[-2, 2]
```

**Critic（状态+动作 → Q值）：**
```
Input: [state, action]拼接
    ↓
Linear(4 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 1)  # 输出Q值
```

### 5.8 结果

<div style="display: flex; gap: 10px; justify-content: center;">
  <img src="_attachments/pendulum_first.gif" style="width:300px; height:300px; object-fit: cover; object-position: bottom;">
  <img src="_attachments/pendulum_last.gif" style="width:300px; height:300px; object-fit: cover;">
</div>

---

## 6 算法进化脉络
```
Q-Learning (1989)
    ├─ 表格法，离散状态
    ├─ 适用：小规模问题
    └─ 优势：简单、可解释
        ↓
DQN (2013)
    ├─ 神经网络拟合Q函数
    ├─ 经验回放 + 目标网络
    ├─ 适用：大规模离散动作
    └─ 突破：Atari游戏超人类
        ↓
DDPG (2015)
    ├─ Actor-Critic分离
    ├─ 连续动作空间
    └─ 适用：机器人控制
        ↓
后续演进：TD3, SAC, PPO...
```

---

## 7 实战技巧总结

### 7.1 调参优先级

**高优先级（影响大）：**
- 学习率 `lr`：太大不收敛，太小学不动
- 折扣因子 `γ`：0.99适合长期任务，0.9适合短期
- 网络结构：先简单后复杂

**中优先级：**
- 批量大小 `batch_size`：32-128通常够用
- 经验池大小：10k-100k
- 目标网络更新频率

**低优先级（微调）：**
- ε衰减率、噪声参数

---

### 7.2 训练不收敛？检查清单

✅ **奖励设计** - 是否太稀疏？考虑reward shaping  
✅ **网络容量** - 太小拟合不了，太大过拟合  
✅ **探索不足** - ε或噪声衰减太快  
✅ **学习率** - 从1e-3开始尝试  
✅ **梯度爆炸** - 加梯度裁剪 `clip_grad_norm_`

---

## 8 常见问题FAQ

**Q1: Q-Learning能否用于连续状态？**  
A: 可以，但需要离散化。状态维度>5时，Q表会很大，建议用DQN

**Q2: DQN为什么不能处理连续动作？**  
A: DQN需要argmax操作选最优动作，连续空间无法枚举所有动作

**Q3: 训练时奖励一直是负数正常吗？**  
A: 正常。很多环境（如Pendulum）的奖励范围本来就是负数，关键看趋势是否上升

**Q4: 何时用on-policy vs off-policy？**  
A: 
- **On-policy**（如PPO）：直接优化当前策略，样本效率低但稳定
- **Off-policy**（如DQN/DDPG）：利用历史数据，样本效率高

**Q5: 我该选哪个算法？**  
A: 
- 离散动作 → DQN/PPO
- 连续动作 → DDPG/SAC/TD3
- 多智能体 → MADDPG/QMIX

---

## 9 代码实现要点

### 9.1 通用结构
```python
# 1. 环境初始化
env = gym.make("CartPole-v1")
state = env.reset()

# 2. 训练循环
for episode in range(episodes):
    state = env.reset()
    for step in range(max_steps):
        # 选择动作
        action = select_action(state)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存储经验
        memory.push(state, action, reward, next_state, done)
        
        # 学习（DQN/DDPG）
        if len(memory) > batch_size:
            learn()
        
        if done:
            break
        state = next_state
```

### 9.2 评估与可视化
```python
# 评估（关闭探索）
@torch.no_grad()
def evaluate(policy, episodes=10):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total = 0
        while True:
            action = policy(state)  # 不加噪声
            state, reward, done = env.step(action)
            total += reward
            if done:
                break
        rewards.append(total)
    return np.mean(rewards)

# 保存训练曲线
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("training_curve.png")
```

---

