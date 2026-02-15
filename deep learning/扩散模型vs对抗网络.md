
面向：快速复习两类经典生成模型的**核心原理、训练/采样流程、优缺点与效果对比。**

---
## 1 概述

- **GAN**：用“生成器 G 造假 + 判别器 D 打假”的对抗博弈，让 G 学到数据分布；**采样快（一前向）**，但训练容易不稳定。

- **DDPM（扩散模型）**：训练时把图片逐步加噪，模型学习“去噪/预测噪声”；生成时从纯噪声逐步还原；**训练更稳、质量常更高**，但**采样慢（多步迭代）**。

---

## 2 GAN 原理与流程

### 2.1 数学目标（经典形式）

GAN 把生成建模写成一个二人零和博弈：

- 生成器：`G(z)` 把随机噪声 `z ~ p(z)` 映射到样本空间，目标“骗过”判别器。

- 判别器：`D(x)` 输出样本 `x` 为真实数据的概率，目标“分清真伪”。

经典目标函数：

  
$$

\min_G \max_D \ \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]

$$


训练时通常交替更新：先更新 D（更会分），再更新 G（更会骗）。

### 2.2 训练直觉

- D 学会区分真/假，相当于为 G 提供“梯度信号”告诉它哪里不像真图。

- 当 G 生成分布逼近真实分布时，D 无法区分，输出接近 0.5。

### 2.3 本项目实现要点（`GAN.py`）

- 数据归一化：`Normalize((0.5,), (0.5,))` 把像素映射到 `[-1, 1]`。

- 模型结构（全连接 MLP）：

  - `Generator`：`z_dim=100 -> ... -> 28*28`，输出用 `Tanh()`，再 reshape 成 `1×28×28`。

  - `Discriminator`：把 `28*28` 输入到 MLP，最后 `Sigmoid()` 输出真概率。

- 损失：`BCELoss()`（二分类交叉熵）

- 小技巧：

  - **Label smoothing**：真样本标签用 `0.9`（不是 1.0），缓解 D 过强导致梯度消失。

  - 训练频率：每轮判别器 1 次，生成器训练 2 次（让 G 追上 D）。

- 可视化保存：把生成结果从 `[-1,1]` 映射回 `[0,1]`：`fake * 0.5 + 0.5`。

### 2.4 常见问题与现象

- **模式崩塌（Mode Collapse）**：G 只会生成少量相似样本（多样性下降）。

- **训练不稳定**：D 太强、梯度消失、震荡。

- **对超参敏感**：学习率、Adam 的 betas、网络容量等都会显著影响效果。

---

## 3 DDPM（扩散模型）原理与流程

### 3.1 前向扩散（加噪）


前向过程把真实图片 `x0` 逐步加噪得到 `x_t`（`t=1..T`）：


$$

q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}x_{t-1}, (1-\alpha_t)I)

$$

常用等价写法（一次采样到任意 t）：

$$

x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\ \epsilon \sim \mathcal{N}(0, I)

$$

其中$$ \bar{\alpha}_t = \prod_{i=1}^t \alpha_i$$
### 3.2 反向生成（去噪）


生成时从 `x_T ~ N(0, I)` 开始，逐步把噪声“还原成图像”。关键是学习一个模型（通常是 UNet）去预测噪声：

- 模型预测：$$\epsilon_\theta(x_t, t)$$
- 用预测的噪声构造 $$`x_{t-1}`$$（带随机项，t>0 时注入噪声）

直觉：模型学会“给你一个带噪图片 + 当前噪声等级 t，我告诉你它混入了多少噪声/该怎么去掉”。

  
### 3.3 本项目实现要点（`diffusion.py`）


- 数据范围：通过 `x * 2 - 1` 映射到 `[-1, 1]`，与 DDPM 训练常见设定一致。

- 步数：`timesteps = 1000`（经典 DDPM 级别，采样开销较大）

- 噪声调度：`cosine_beta_schedule`（余弦 schedule，通常比线性更平滑）

- 模型：简化版 **UNet + 时间嵌入**

  - `timestep_embedding`：用 `sin/cos` 形成时间步的 embedding

  - `ResidualBlockWithTime`：残差块里把时间 embedding 通过 `Linear` 投影并加到特征图上，实现“时间注入”

  - 下采样 `MaxPool2d(2)`，上采样 `Upsample(scale_factor=2)`，并用 skip connection 拼接

- 训练目标：最小化 `MSE(pred_noise, true_noise)`（预测噪声的均方误差）

- 采样：从高斯噪声出发，`for t in reversed(range(timesteps))` 逐步去噪

- 保存：每 10 epoch 生成 64 张并拼成 8×8 网格保存到 `ddpm_results/`

  
### 3.4 常见问题与现象

  
- **采样很慢**：1000 步反向迭代；如果想更快需要 DDIM/减少 steps 等。

- **训练更稳**：优化目标是 MSE 回归，相比 GAN 的对抗更可控。

- **质量与细节**：在相同任务上通常更容易得到更清晰、更多样的结果（代价是采样计算量）。

  
---
## 4 效果对比图
左：对抗网络；右：扩散模型

<div style="display: flex; gap: 10px; justify-content: center;">
  <img src="_attachments/GAN-epoch10.png" style="width:300px; height:300px; object-fit: cover; object-position: bottom;">
  <img src="_attachments/diffusion-epoch10.png" style="width:300px; height:300px; object-fit: cover;">
</div>


<div style="display: flex; gap: 10px; justify-content: center;">
  <img src="_attachments/GAN-epoch100.png" style="width:300px; height:300px; object-fit: cover; object-position: bottom;">
  <img src="_attachments/diffusion-epoch100.png" style="width:300px; height:300px; object-fit: cover;">
</div>
---
## 5 效果对比表

| 维度         | GAN              | DDPM/扩散             |
| ---------- | ---------------- | ------------------- |
| 训练稳定性      | 较差，容易震荡/崩塌       | 较好，目标函数更“平滑”        |
| 采样速度       | 快（一次前向）          | 慢（多步去噪；本项目 1000 步）  |
| 生成质量（常见经验） | 取决于技巧与调参，可能很强但不稳 | 通常更稳、更细腻、更高质量       |
| 多样性        | 可能 mode collapse | 通常较好（仍依赖模型容量/训练）    |
| 训练成本       | 相对低（小模型即可玩）      | 相对高（UNet + 多步/更大算力） |
| 难点         | 对抗平衡、梯度问题、模式崩塌   | 采样加速、步数/调度、模型容量     |

在 MNIST 这种简单数据上：

- GAN 往往很快能看到“像数字”的结果，但可能出现数字类型不均衡或重复。

- 扩散模型训练收敛后通常更均匀、轮廓更稳定，但训练与采样时间更长。
---

## 5 如何评价“效果好不好”（建议指标）


本项目主要用“肉眼看网格图”做定性对比。若要更学术/工程化：

- **FID**（越低越好）：衡量生成分布与真实分布的距离（常用且更可信）。

- **IS**（越高越好）：更偏分类器信心与多样性（对数据/分类器依赖较强）。

- MNIST 也可做：用预训练分类器计算生成样本的分类分布、覆盖率与置信度。

---

## 6 本项目可进一步改进的方向（可选）


### 6.1 GAN 改进（更稳更强）

- 损失：换成 WGAN / WGAN-GP（更稳定）

- 结构：卷积 DCGAN、谱归一化（Spectral Norm）

- 训练技巧：TTUR、更多正则、数据增强（DiffAugment）

### 6.2 扩散模型改进（更快更好）


- 采样加速：DDIM、减少 steps、或用更好的 sampler

- 结构增强：更标准的 UNet（多层、多尺度 attention）

- 条件生成：加类别条件（MNIST label conditioning）

---

