
---
## 1 整体视角

1. **句子分类（情感）**：`text -> tokenizer -> encoder -> logits -> softmax -> label`

2. **序列到序列（翻译）**：`text -> tokenizer -> encoder-decoder -> generate -> decode`

3. **自回归生成（续写）**：`prompt -> tokenizer -> decoder-only -> sampling -> decode`

它们共同点是：**先把文本映射成 token，再把模型输出转成可解释的结果**（概率/翻译文本/生成文本）。

---
## 2 NLP核心概念：Tokenizer与模型架构

> 理解文本如何被模型"看见"和"生成" 

---
### 2.1 核心流程图
```
文本输入 
  ↓ tokenizer切分
Token序列 ["I", "love", "AI"] 
  ↓ 查词表
Token ID [101, 1234, 5678]
  ↓ 查Embedding表
向量矩阵 [[0.1,-0.3,...], [0.5,0.2,...], ...]，形状通常是 [词表大小, 向量维度]，由训练而来
  ↓ Transformer计算
输出结果（分类/生成）
```
附：为了平衡“词典大小”和“语义表达”，现多采用子词为单位，unhappiness-->un+happi+ness
Embedding 表是一个巨大的权重矩阵 [词表大小, 向量维度]

---
### 2.2 Tokenizer详解：文本如何变成数字

#### 2.2.1 什么是Token？

Token = 文本的最小处理单元（不一定是完整单词）

例如："unbelievable" 可能被切成：
- `un` + `believ` + `able`（子词切分）
- 目的：平衡词表大小和覆盖未登录词（OOV）

#### 2.2.2 完整流程

##### 2.2.2.1 Step 1: 文本规范化
- 统一大小写、去除特殊符号等
- 不同tokenizer策略不同（有的保留大小写）

#### 2.2.3 Step 2: 子词切分（Subword Tokenization）

三种主流算法：

| 算法 | 代表模型 | 特点 |
|------|----------|------|
| **BPE** | GPT-2 | 合并高频字符对，字节级处理 |
| **WordPiece** | BERT/RoBERTa | 用`##`标记子词（如`play##ing`） |
| **SentencePiece** | MarianMT | 用`▁`表示词边界/空格 |

#### 2.2.4 Step 3: Token → ID（查词表）
```python
# 示例
vocab = {"love": 1234, "AI": 5678}
tokens = ["love", "AI"]
ids = [1234, 5678]  # 查表得到
```

#### 2.2.5 Step 4: ID → Embedding向量
```python
# 模型内部有个大矩阵 E (词表大小 × 向量维度)
E = torch.randn(50000, 768)  # 5万词汇，每个768维
embedding = E[1234]  # 用ID索引取向量
```

#### 2.2.6 Step 5: 加位置编码 → 进Transformer

**关键点：** Token本身只是字符串片段，真正进模型的是**向量**

---

## 3 三种模型架构对比

### 3.1 架构1: Encoder-Only（编码器）

**代表模型：** BERT、RoBERTa  
**适用场景：** 分类、情感分析、命名实体识别

**工作流程：**
```
输入: "这电影真棒"
  ↓ Tokenizer
Token IDs: [101, 1234, 5678, 9012, 102]
  ↓ Encoder (Transformer)
隐状态矩阵 H (每个token一个向量)
  ↓ Pooling (取[CLS]位置)
句子向量
  ↓ 分类头（线性层）
输出: [0.1, 0.9] → 正面情感
```
**特点**：
- ✅ 双向注意力：每个token能看到前后所有token
- ✅ 理解能力强，适合分类任务
- ❌ 不能直接生成文本

---

### 3.2 架构2: Encoder-Decoder（编码-解码）

**代表模型：** T5、BART、MarianMT（翻译）  
**适用场景：** 翻译、摘要、问答

**工作流程：**
```
源语言: "I love AI"
  ↓ Encoder编码
隐状态 H_source
  ↓ 传递给Decoder
  ↓ Decoder生成（逐个token）
目标语言: "我" → "爱" → "AI" → [EOS]
```

**核心机制 - Cross-Attention：**
- Decoder每生成一个词，都会"回头看"Encoder的隐状态
- 像翻译时不断对照原文

**训练 vs 推理的区别：**
- 训练时： Teacher Forcing（用真实目标序列，学得快）
- 推理时： 只能用自己生成的前缀（可能累积错误）

---

### 3.3 架构3: Decoder-Only（仅解码器）

**代表模型：** GPT-2、GPT-3、LLaMA  
**适用场景：** 文本续写、对话、代码生成

**工作流程：**
```
输入提示: "从前有座山，"
  ↓ Decoder预测下一个词
"山" (概率最高)
  ↓ 追加到输入
"从前有座山，山"
  ↓ 继续预测
"里" → "有" → "座" → "庙" ...
```

**特点：**
- **只能看到左边的token（单向注意力）**
- **自回归生成：P(下一个词 | 之前所有词)**
- 最灵活，可以做各种生成任务

---

## 4 关键参数详解

### 4.1 padding_side（填充方向）

**为什么需要padding？**  
- Batch推理时，句子长度不一致
- 需要填充到统一长度
```python
# 左填充 vs 右填充
sentences = ["Hello", "Hi there"]

# padding_side="right" (常用于BERT)
# [101, 7592, 102, 0, 0]      # Hello + padding
# [101, 7632, 2045, 102, 0]   # Hi there + padding

# padding_side="left" (常用于GPT生成)
# [0, 0, 101, 7592, 102]      # padding + Hello
# [0, 101, 7632, 2045, 102]   # padding + Hi there
```

**影响：** Decoder-only生成时，左填充更合理（保证最后位置是真实内容）

---

### 4.2 attention_mask（注意力掩码）

**作用：** 告诉模型哪些位置是padding，应该被忽略
```python
input_ids      = [101, 1234, 5678, 0, 0]
attention_mask = [  1,    1,    1, 0, 0]
#                   ↑ 真实token    ↑ padding
```

**不传会怎样？** 模型会把padding当真实内容处理 → 结果不稳定

---

### 4.3 pad_token_id

**问题：** GPT-2等模型默认没有`<PAD>`标记

**常见做法：**
```python
tokenizer.pad_token = tokenizer.eos_token
# 把结束符当填充符用（但要配合attention_mask）
```

---

## 5 实战对应关系

| 脚本文件 | 模型 | Tokenizer类型 | 特征 |
|---------|------|--------------|------|
| NLP情感分析.py | RoBERTa | WordPiece | 看到`##`前缀 |
| NLP翻译模型.py | MarianMT | SentencePiece | 看到`▁`符号 |
| NLP文本生成.py | GPT-2 | Byte-level BPE | 看到`Ġ`符号 |

---

## 6 常见问题FAQ

**Q1: 为什么要用子词而不是完整单词？**  
A: 控制词表大小（完整单词可能有几十万），同时处理未登录词（如新词、拼写错误）

**Q2: Token数量是越少越好吗？**  
A: 不一定。太少信息丢失，太多计算成本高。通常平衡在512-2048之间

**Q3: 中文和英文tokenizer有什么区别？**  
A: 中文常用字符级或词级切分，英文用子词级。中文token数量通常更多

**Q4: 如何查看文本被切成了什么token？**  
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokens = tokenizer.tokenize("我爱自然语言处理")
print(tokens)  # ['我', '爱', '自然', '语言', '处理']
```                     

---
## 7 情感分析（二分类）——`NLP情感分析.py`

### 7.1 思路

- 直接用 Hugging Face 上已经微调好的中文二分类情感模型：`uer/roberta-base-finetuned-jd-binary-chinese`

- 推理输出是 `logits=[neg,pos]`，做一次 `softmax` 得到概率

- 用 `P(pos)` 和 `P(neg)` 的大小决定标签
### 7.2 关键细节（易忽略）

- **截断与 padding**：批量推理时必须 `padding=True, truncation=True`，否则张量无法堆叠

- **设备选择**：CPU/GPU/MPS 自动选（脚本里有 `pick_device()`）

- **结果可解释性（轻量版）**：脚本里做了一个 *token saliency*（`|grad * emb|`）来大致看模型更“盯着”哪些 token

  - 这不是严格的可解释方法（如 Integrated Gradients），但学习阶段足够直观

### 7.3 如何判断“模型是否靠谱”

- 先用“强情绪句”测试：明显好评/差评是否分对

- 再用“模棱两可句”测试：概率是否接近（而不是瞎自信）

- 最后再考虑引入自己的领域数据做微调
---
## 8 情感分析（三分类）——`NLP情感分析三分类.py`

### 8.1 现有脚本在做什么

脚本思路是：

- 仍然加载二分类模型 `uer/roberta-base-finetuned-jd-binary-chinese`

- 强行把输出层改成 3 类（`num_labels=3`，并重置 `classifier`）

### 8.2 重点：这样不能直接得到“三分类能力”

把 `num_labels` 从 2 改成 3 只是在结构上变成 3 维输出；分类头参数是随机初始化的，没有三分类训练数据就不会有可靠效果。

想要三分类（正/负/中性），通常有两条正确路线：

1. 直接选用三分类已微调模型（省事，效果取决于模型与数据域是否匹配）

2. 自己微调三分类（最稳：数据、标签、域都可控）

### 8.3 学习阶段的可用 baseline（不训练也能跑）

如果手头只有二分类模型，可以做一个很常用的“阈值中性”基线：

- 先算 `P(pos)` 与 `P(neg)`

- 如果 `|P(pos) - P(neg)| < threshold`，判为 **中性**

- 否则取概率更大的那一类

这不是严格三分类，但能把“犹豫样本”从正/负里分出来，适合作为学习与对比的 baseline。

---
## 9 翻译（MarianMT）——`NLP翻译模型.py`

### 9.1 思路

- MarianMT 是典型的 encoder-decoder（seq2seq）翻译模型

- 核心步骤：

  1. `tokenizer(text)` 得到模型输入

  2. `model.generate(...)` 生成目标语言 token

  3. `tokenizer.decode(...)` 得到可读文本
### 9.2 关键细节

- **方向一定要核对**：`Helsinki-NLP/opus-mt-en-zh` 是 英文 -> 中文（en→zh）

- 推理参数：学习阶段先用默认 `generate()`；想控制风格再加 `num_beams`、`max_new_tokens` 等
---
## 10 文本生成（GPT-2）——`NLP文本生成.py`

### 10.1 思路

  
- GPT-2 是 decoder-only 自回归模型：一次预测下一个 token

- 生成就是不断“把自己刚生成的 token 再喂回去”

### 10.2 关键细节（决定生成质量）

- `temperature`：越小越保守，越大越发散

- `top_k`/`top_p`：采样空间裁剪（常用组合：`top_p` + `temperature`）

- `no_repeat_ngram_size`：抑制复读（但太大可能会伤害连贯性）

- 可复现性：需要设置随机种子，否则每次生成都不一样
---
## 11 本次运行结果（2026-02-15）

### 11.1 情感（二分类 + 三分类 baseline，节选）

```

这手机续航太顶了，屏幕也很舒服，真香。

二分类：正向 | P(pos)=0.994 P(neg)=0.006 | 三分类baseline：正向

  

客服态度很差，物流慢得离谱，太失望了。

二分类：负向 | P(pos)=0.015 P(neg)=0.985 | 三分类baseline：负向

  

感觉一般。

二分类：负向 | P(pos)=0.484 P(neg)=0.516 | 三分类baseline：中性（阈值策略把“不确定”单独分出来）

  

Token saliency（对“客服态度很差...”这句，Top8）：

慢(0.0146) 流(0.0114) 谱(0.0100) 服(0.0089) 客(0.0087) 态(0.0082) 差(0.0081) 望(0.0080)

```
### 11.2 翻译（en→zh，节选）

```

EN: I love you, but I don't know if it's going to be you or me.

ZH: 我爱你 但我不知道是你还是我

  

EN: This project is small, but it helps me learn the full inference pipeline.

ZH: 这个项目很小,但它能帮助我 学习完整的推论管道。

```
### 11.3 GPT-2 生成（节选）

```

Prompt: I love you

Gen   : I love you, Daddy," I said.

        "Yes," he replied. "I know."

```

