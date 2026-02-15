
> 目标：用小规模对话数据，把通用聊天基座微调成“更像某个朋友/人设”的聊天风格，并用可复现的方式做对比评估。

> 本仓库相关脚本：训练 `llm.py`、数据清洗 `clear_data.py`、聊天推理 `chat.py`、单人对比 `compare.py`、多人对比 `multi_compare.py`。

> 运行产物（已生成）：`compare_results.md`（base vs 单个 LoRA），`multi_compare_results.md`（base vs 3 personas）。
---
## 1 技术路线总览（做什么 & 为什么这么做）

这套方案可以理解为一条“性价比优先”的微调流水线：

1. **SFT（监督微调）**：用成对的「用户输入 → 助手输出」让模型学习“怎么回答”和“以什么语气回答”。

2. **LoRA（低秩适配）**：不改动全量参数，只训练一小部分“可插拔”的适配器权重，把训练成本和保存体积大幅压缩。

3. **4bit 量化加载（nf4）**：训练/加载时把大模型权重压到 4bit，进一步降低显存压力（配合 bf16/fp16 计算）。

最终你保存的不是完整基座权重，而是：

- LoRA 适配器（`adapter_model.safetensors` + `adapter_config.json`）

- tokenizer / chat template（用于保证训练与推理提示词格式一致）

推理时：基座模型 + 某个 LoRA 适配器 组合在一起得到最终风格。

---
## 2 实现思路（从数据到模型的完整闭环）

### 2.1 数据如何塑造“人设/风格”

SFT 的本质是“模仿分布”：

- 数据里出现的口癖、句式、情绪倾向，会被模型学到并放大。

- 数据里出现的噪声（空回复、乱码、截断、无意义短答），也会直接体现在生成上（例如回答过短、答非所问）。

因此，数据侧通常要保证两件事：

- **结构正确**：每条样本都能稳定映射为「system/user/assistant」对话格式。

- **质量可控**：对坏样本做过滤，对过短/空洞输出做约束或补充。

仓库里数据采用 JSONL：每行一个对象，核心字段是 `instruction` 和 `output`。

### 2.2 Prompt / Chat Template：训练与推理必须同构

“模板一致”是风格微调是否有效的关键：

- 训练时：system + user + assistant（assistant 给出标准答案）

- 推理时：system + user +（assistant 作为 generation 起点）

如果模板不一致，常见表现是：

- 微调“像没生效”（风格不明显）

- 角色不稳定（system 约束不起作用）

- 把用户话当成要生成的内容（结构错位）

本项目通过 tokenizer 的 `apply_chat_template` 来统一模板，避免手写 prompt 时的细节偏差。

### 2.3 LoRA：改动最小但影响最大的“关键层”

LoRA 通常注入在注意力投影与 MLP 投影层，因为它们对：

- 语气风格（语调、口癖、礼貌程度）

- 组织方式（长短、分点、解释倾向）

影响非常明显，而参数量又可控。

实践上需要关注两点：

- **模块名是否匹配**（不同模型架构命名不同；不匹配就等于没注入）

- **LoRA 容量与过拟合**（`r` 越大越能拟合风格，也更容易把训练集口癖“背下来”）

### 2.4 4bit + 分页优化器：让单卡4060也能跑得动

对于 1~3B 级别模型，常见瓶颈是显存：

- 4bit 量化用于“存储侧降本”

- bf16/fp16 计算用于“算力侧折中”

- 分页优化器用于降低优化器状态占用

你获得的是：更低门槛的本地训练/复现实验能力，但代价是：

- 平台兼容性（Windows 的 bitsandbytes 可能更敏感）

- 数值行为更依赖版本组合（transformers/trl/peft/bnb）
---
## 3 训练与推理：工程化落地怎么做

### 3.1 训练流水线（概念步骤）

1. 数据清洗（过滤坏样本，保证结构一致）

2. 加载基座模型（可选 4bit 量化以省显存）

3. 注入 LoRA（只训练适配器参数）

4. 用 SFT 跑训练（注意模板、学习率、epoch，避免过拟合）

5. 导出 LoRA 适配器与 tokenizer（形成可插拔的 persona）
### 3.2 推理与“记忆”的实现思路

`chat.py` 的“记忆”属于最简单可控的一类：把最近 N 轮对话直接拼到上下文。

- 优点：实现简单，可解释，便于复现。

- 局限：上下文线性增长，长对话会截断；只按“最近”取，不按“相关”取。

升级方向（只写思路）：

- **摘要记忆**：把更早对话压缩成 summary，保留长期信息。

- **检索记忆**：embedding + top-k 取相关历史，再与近期对话拼接。
---
## 4 评估：怎么证明“微调确实改变了模型”

评估最怕“凭感觉”。这里建议遵循三条原则：

1. **固定 prompts**：覆盖情绪、建议、吐槽、简短回复等场景。

2. **固定生成参数**：温度、top_p、最大长度等都固定，才可对比。

3. **同屏对照**：base vs tuned 并排展示，差异才清晰。

本仓库已把结果写成可阅读的 Markdown 报告（建议直接打开看全量）：
  
- base vs 单个 LoRA：`compare_results.md`

- base vs 三个 persona：`multi_compare_results.md`

### 4.1 节选：base vs 单个 LoRA

下面只放几条最能体现差异的节选（完整请看 `compare_results.md`）。
#### 4.1.1 Prompt：我今天又拖延了，一下午啥也没干，心态有点崩。

**Base（微调前，节选）**

```

听到你最近感到拖延和心态崩溃，我非常理解你的感受。

...

1. 设定明确的目标：...

2. 制定计划：...

3. 避免分心：...

```

**Tuned（微调后）**

```

是呀，我也是。

```

> 读法：base 更像“标准助手/说明文”；tuned 更像“朋友口吻/陪聊”，但也更容易信息量不足——这通常是数据与训练策略的 trade-off。
### 4.2 节选：三人 LoRA（base vs girl/jiangjian/peixuan）

仓库里当前可见三份风格数据与对应 LoRA：

- `girl`：`girl_cleaned.jsonl` + `girl_qwen_chat_1.8b/`

- `jiangjian`：`jiang_jian_style.jsonl` + `jiangjian_qwen_chat_1.8b/`

- `peixuan`：`peixuan.jsonl` + `peixuan_qwen_chat_1.8b/`

完整对比请看 `multi_compare_results.md`。这里节选一条：
#### 4.2.1 Prompt：我准备考研但很焦虑，给我一个今晚能做的最小行动（60字以内）。

**Base**

```

今晚花半小时读研参考书，做好笔记和整理思路，缓解焦虑情绪。...

```

**girl**

```

躺到现阶段状态能接受的最好的作息时间。

```

**jiangjian**

```

背一个今天不会的单词。

```

**peixuan**

```

洗个澡吧

```

---
## 5 复现指南（尽量少踩坑）

### 5.1 依赖/版本的核心注意点

这类项目最常见的不稳定因素不是“原理”，而是“版本组合”：

- TRL 的 `SFTTrainer` 参数/行为可能变

- 模型仓库的 chat template/remote code 细节可能变

- 量化/分页优化器对系统与驱动更敏感

如果你只想复现对比结果：优先跑 `compare.py` / `multi_compare.py`，先把 base 与 LoRA 的差异跑通，再回头优化训练配置。
### 5.2 推荐的最小命令（按闭环走）
```powershell

# 1) 清洗

python .\clear_data.py

  

# 2) 训练（需要能加载到基座模型）

python .\llm.py

  

# 3) base vs 单个 LoRA 对比（会写 compare_results.md）

python .\compare.py

  

# 4) base vs 3 人对比（会写 multi_compare_results.md）

python .\multi_compare.py

```
---
## 6 常见问题与改进方向（只讲思路）

### 6.1 tuned 回答过短/空洞

原因往往来自数据分布：训练集中短答比例高，或者输出包含大量“语气词/省略号”。

改进思路：

- 数据侧补充“高信息密度”的示例

- 训练侧降低 epoch / 调整学习率，缓解背训练集口癖

- 推理侧加入长度/结构约束（例如要求“用 2~3 句回答”）

### 6.2 风格不稳定 / 像没微调

优先检查“模板一致性”：训练时和推理时的 system/user/assistant 结构是否同构，tokenizer 的 chat template 是否一致加载。

### 6.3 多 persona 的工程化

多 persona 的关键是“可插拔”：

- 每个 persona 一套 LoRA adapter

- 同一基座模型上加载多个 adapter，通过切换 adapter 选择风格

这也是 `multi_compare.py` 的实现思路：同一 base + 多个 adapter 做并列对比。