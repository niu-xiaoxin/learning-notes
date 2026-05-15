# Building makemore Part 4: Becoming a Backprop Ninja

> **课程来源**: [Andrej Karpathy - YouTube](https://www.youtube.com/watch?v=q8SA3rM6ckI)
>
> **核心目标**: 在 tensor 层级手写整个神经网络的反向传播，替代 `loss.backward()`，深入理解梯度的流动机制。

---

## 目录

- [1. 为什么要手写反向传播](#1-为什么要手写反向传播)
- [2. 历史小插曲](#2-历史小插曲)
- [3. Starter Code 介绍](#3-starter-code-介绍)
- [4. Exercise 1：逐 tensor 手写反向传播](#4-exercise-1逐-tensor-手写反向传播)
  - [4.1 dlogprobs：起点](#41-dlogprobs起点)
  - [4.2 dprobs：穿过 log](#42-dprobs穿过-log)
  - [4.3 dcounts_sum_inv：广播反向 = sum](#43-dcounts_sum_inv广播反向--sum)
  - [4.4 dcounts：变量被用两次 → 梯度累加](#44-dcounts变量被用两次--梯度累加)
  - [4.5 dnorm_logits / dlogit_maxes / dlogits：穿过 exp、减法、max](#45-dnorm_logits--dlogit_maxes--dlogits穿过-exp减法max)
  - [4.6 穿过第二个线性层（矩阵乘法反向传播）](#46-穿过第二个线性层矩阵乘法反向传播)
  - [4.7 穿过 tanh 激活函数](#47-穿过-tanh-激活函数)
  - [4.8 穿过 BatchNorm 的 scale 和 shift](#48-穿过-batchnorm-的-scale-和-shift)
  - [4.9 穿过 BatchNorm 标准化内部](#49-穿过-batchnorm-标准化内部)
  - [4.10 穿过第一个线性层](#410-穿过第一个线性层)
  - [4.11 穿过 view 和 embedding 查表](#411-穿过-view-和-embedding-查表)
- [5. Exercise 2：cross-entropy 解析反向](#5-exercise-2cross-entropy-解析反向)
- [6. Exercise 3：BatchNorm 解析反向](#6-exercise-3batchnorm-解析反向)
- [7. Exercise 4：全部整合，替代 loss.backward()](#7-exercise-4全部整合替代-lossbackward)
- [8. 全课总结](#8-全课总结)
- [附录 A：反向传播规则速查表](#附录-a反向传播规则速查表)
- [附录 B：完整反向传播代码](#附录-b完整反向传播代码)

---

## 1. 为什么要手写反向传播

Andrej 认为 `loss.backward()` 虽然方便，但反向传播是一个 **leaky abstraction（漏水的抽象）**——你不能把可微的"乐高积木"随意拼在一起然后祈祷一切正常。如果不理解内部机制，它会以各种微妙的方式失灵。

典型"翻车"场景：

- **激活函数饱和区（flat tails）**：tanh / sigmoid 在两端梯度接近 0，梯度会"死掉"
- **Dead neurons**：ReLU 永远输出 0
- **梯度爆炸 / 梯度消失**：RNN 中尤其常见
- **隐蔽 bug**：有人想裁剪 loss 最大值 `loss = torch.clamp(loss, max=10)`，但实际效果是把异常样本的梯度设为了 0，导致离群样本被完全忽略

核心观点：**PyTorch 有 autograd ≠ 你可以不懂反向传播。** 在 micrograd 中我们在标量层级手写过，这次在 tensor 层级做，是升级版。

---

## 2. 历史小插曲

大约 10 年前，手写反向传播是行业标准操作：

- **2006 年**：Hinton & Salakhutdinov 的 Science 论文，训练 RBM（受限玻尔兹曼机），当时甚至不一定用反向传播，而是用 Contrastive Divergence
- **2010 年**：Andrej 自己用 MATLAB 写 RBM 训练代码，全部手写梯度，在 CPU 上跑
- **2014 年**：Andrej 的论文代码（Python + numpy），手写 backward pass + gradient checker

历史演变路线：**MATLAB → numpy → PyTorch**。当年的 gradient checker（用有限差分数值近似验证解析梯度）就是今天我们的 `cmp` 函数 + PyTorch autograd。

---

## 3. Starter Code 介绍

### 3.1 cmp 函数

```python
def cmp(s, dt, t):
    ex  = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff:.6e}')
```

三个指标：exact（完全相等）、approximate（近似相等，允许浮点误差）、maxdiff（最大绝对差值）。

### 3.2 前向传播的"展开"

Andrej 故意把前向传播写得非常啰嗦，把每一步拆成独立变量，是为了**让我们能对每一步单独求梯度**。

完整前向传播代码：

```python
# Embedding
emb = C[Xb]                                          # (32, 3, 10)
embcat = emb.view(emb.shape[0], -1)                   # (32, 30)

# Linear Layer 1
hprebn = embcat @ W1 + b1                             # (32, 64)

# BatchNorm
bnmeani = hprebn.sum(0, keepdim=True) / n             # (1, 64)
bndiff = hprebn - bnmeani                             # (32, 64)
bndiff2 = bndiff**2                                   # (32, 64)
bnvar = bndiff2.sum(0, keepdim=True) / (n - 1)       # (1, 64)  Bessel's correction
bnvar_inv = (bnvar + 1e-5)**(-0.5)                    # (1, 64)
bnraw = bndiff * bnvar_inv                            # (32, 64)
hpreact = bngain * bnraw + bnbias                     # (32, 64)

# Activation
h = torch.tanh(hpreact)                               # (32, 64)

# Linear Layer 2
logits = h @ W2 + b2                                  # (32, 27)

# Cross-Entropy Loss (展开版)
logit_maxes = logits.max(1, keepdim=True).values      # (32, 1)
norm_logits = logits - logit_maxes                     # (32, 27)
counts = norm_logits.exp()                            # (32, 27)
counts_sum = counts.sum(1, keepdim=True)              # (32, 1)
counts_sum_inv = counts_sum**(-1)                     # (32, 1)
probs = counts * counts_sum_inv                       # (32, 27)
logprobs = probs.log()                                # (32, 27)
loss = -logprobs[range(n), Yb].mean()                 # scalar
```

### 3.3 参数表

| 变量 | Shape | 说明 |
|------|-------|------|
| `C` | (27, 10) | 字符 embedding 查找表 |
| `W1` | (30, 64) | 第一层权重 |
| `b1` | (64,) | 第一层 bias |
| `W2` | (64, 27) | 第二层权重 |
| `b2` | (27,) | 第二层 bias |
| `bngain` | (1, 64) | BatchNorm 缩放参数 |
| `bnbias` | (1, 64) | BatchNorm 偏移参数 |

---

## 4. Exercise 1：逐 tensor 手写反向传播

从 `loss` 开始，逆着前向传播的方向，一步步推到最开头的 `C`。

### 4.1 dlogprobs：起点

前向：`loss = -logprobs[range(n), Yb].mean()`

这里做了三件事：**索引（挑选）**、**取负**、**求平均**。

- 只有被正确类别选中的位置有梯度（其他位置为 0）
- 取负 → 梯度带负号
- 求平均 → 梯度 = `-1/n`

```python
dlogprobs = torch.zeros_like(logprobs)                # (32, 27) 全零
dlogprobs[range(n), Yb] = -1.0 / n                    # 只在正确类别位置填 -1/n
```

### 4.2 dprobs：穿过 log

前向：`logprobs = probs.log()`

$\frac{d}{dx} \log(x) = \frac{1}{x}$

```python
dprobs = (1.0 / probs) * dlogprobs                    # (32, 27)
```

### 4.3 dcounts_sum_inv：广播反向 = sum

前向：`probs = counts * counts_sum_inv`

`counts` 是 `(32, 27)`，`counts_sum_inv` 是 `(32, 1)`。在前向中 `counts_sum_inv` 被水平广播了 27 次。

**核心规则：广播（复制）的反向 = 沿复制方向求和（sum）。**

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)  # (32, 1)
```

为什么是这样？如果一个值被复制到了 27 个位置，那 27 个位置各自收到的梯度全部要累加回来。

### 4.4 dcounts：变量被用两次 → 梯度累加

`counts` 在前向中出现了两次：
1. `probs = counts * counts_sum_inv`
2. `counts_sum = counts.sum(1, keepdim=True)`

**核心规则：一个变量被用多次 → 梯度必须 `+=`（累加），不是 `=`（覆盖）。**

```python
# 第一条路：来自 probs
dcounts = counts_sum_inv * dprobs                      # (32, 27)

# 第二条路：穿过 counts_sum → counts_sum_inv
dcounts_sum = (-counts_sum**(-2)) * dcounts_sum_inv    # (32, 1)
dcounts += torch.ones_like(counts) * dcounts_sum       # 广播 + 累加
```

`counts.sum()` 的反向就是把梯度复制回去（sum 的反向 = 广播）。

### 4.5 dnorm_logits / dlogit_maxes / dlogits：穿过 exp、减法、max

**穿过 exp**：$\frac{d}{dx} e^x = e^x$，本地梯度就是 `counts` 本身。

```python
dnorm_logits = counts * dcounts                        # (32, 27)
```

**穿过减法**：`norm_logits = logits - logit_maxes`

```python
dlogits = dnorm_logits.clone()                         # logits 的第一部分梯度
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)    # (32, 1)，广播反向 = sum
```

**穿过 max**：`logit_maxes = logits.max(1, keepdim=True).values`

max 操作只有被选中的那个位置（最大值位置）有梯度 = 1，其他位置 = 0。用 one-hot 编码表示。

```python
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]).float() * dlogit_maxes
```

**但实际上**：`logit_maxes` 只是为了数值稳定，它完全不影响 softmax 的输出。所以 `dlogit_maxes` 这一路的梯度贡献理论上是 0。Andrej 还是写了出来以保持形式完整。

### 4.6 穿过第二个线性层（矩阵乘法反向传播）

前向：`logits = h @ W2 + b2`

**推导方法**：写一个 2×2 的小例子，展开成标量运算（全退化成 micrograd），逐元素求偏导，再排回矩阵，发现结果就是矩阵乘法 + 转置。

**三条公式**（对于 `D = A @ B + C`）：

| 梯度 | 公式 | 说明 |
|------|------|------|
| `dA` | `dD @ B.T` | 上游梯度 × 另一方的转置 |
| `dB` | `A.T @ dD` | 另一方的转置 × 上游梯度 |
| `dC` | `dD.sum(0)` | 广播反向 = sum |

**Andrej 的"秘密"：不用记公式，用 shape 匹配推导！**

以 `dh` 为例：
1. `dh` 的 shape 必须 = `h` 的 shape = `(32, 64)`
2. 一定是 `dlogits` 和 `W2` 的某种矩阵乘法
3. `dlogits` 是 `(32, 27)`，`W2` 是 `(64, 27)`
4. 唯一能得到 `(32, 64)` 的方式：`(32, 27) @ (27, 64)` → 需要 `W2.T`

```python
dh  = dlogits @ W2.T             # (32, 27) @ (27, 64) → (32, 64)
dW2 = h.T @ dlogits              # (64, 32) @ (32, 27) → (64, 27)
db2 = dlogits.sum(0)             # (32, 27) → (27,)
```

### 4.7 穿过 tanh 激活函数

前向：`h = torch.tanh(hpreact)`

tanh 的导数有一个友好的等价形式：

$$\text{如果 } a = \tanh(z)，\text{则 } \frac{da}{dz} = 1 - a^2$$

**注意**：$1 - a^2$ 中的 $a$ 是 tanh 的**输出**（不是输入），所以直接用已有的 `h`。

```python
dhpreact = (1.0 - h**2) * dh     # (32, 64)
```

**饱和区的直觉**：当 `h → ±1` 时，`1 - h² → 0`，梯度被杀死——这就是梯度消失的微观机制，也是 Part 3 里正确初始化如此重要的原因。

### 4.8 穿过 BatchNorm 的 scale 和 shift

前向：`hpreact = bngain * bnraw + bnbias`

这里的乘法是**逐元素乘法**（不是矩阵乘法），比线性层简单。但要注意 `bngain (1, 64)` 和 `bnbias (1, 64)` 有广播。

```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)   # (1, 64)
dbnraw  = bngain * dhpreact                          # (32, 64)
dbnbias = dhpreact.sum(0, keepdim=True)              # (1, 64)
```

规律：**被广播的那一方 → 反向 sum + keepdim；没被广播的 → 直接逐元素乘。**

`bngain` 和 `bnbias` 是可学习参数，梯度到此为止。`bnraw` 还要继续往上游传。

### 4.9 穿过 BatchNorm 标准化内部

这是 Exercise 1 中最长最难的一段。前向代码：

```python
bnmeani   = hprebn.sum(0, keepdim=True) / n          # 均值 μ       (1, 64)
bndiff    = hprebn - bnmeani                          # x - μ        (32, 64)
bndiff2   = bndiff**2                                 # (x - μ)²     (32, 64)
bnvar     = bndiff2.sum(0, keepdim=True) / (n - 1)   # 方差 σ²      (1, 64)
bnvar_inv = (bnvar + 1e-5)**(-0.5)                    # 1/√(σ²+ε)   (1, 64)
bnraw     = bndiff * bnvar_inv                        # 标准化结果    (32, 64)
```

**难点**：`bndiff` 分两条路（→ `bnraw` 和 → `bndiff2`），`hprebn` 也分两条路（→ `bndiff` 和 → `bnmeani`），梯度要 `+=` 汇合。

**新的对偶规律**：前向 sum ⟷ 反向广播，前向广播 ⟷ 反向 sum。

逐步推导：

```python
# Step 1: bnraw = bndiff * bnvar_inv → 求 dbndiff(第一部分) 和 dbnvar_inv
dbndiff    = bnvar_inv * dbnraw                                     # (32, 64)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)                 # (1, 64)

# Step 2: bnvar_inv = (bnvar + 1e-5)**(-0.5) → 幂规则
dbnvar = (-0.5) * (bnvar + 1e-5)**(-1.5) * dbnvar_inv              # (1, 64)

# Step 3: bnvar = bndiff2.sum(0, keepdim=True) / (n-1)
# 前向 sum → 反向广播
dbndiff2 = (1.0 / (n - 1)) * dbnvar                                # 广播 (1,64) → (32,64)

# Step 4: bndiff2 = bndiff**2 → d/dx x² = 2x
dbndiff += 2 * bndiff * dbndiff2                      # ⚠️ += 累加第二部分

# Step 5: bndiff = hprebn - bnmeani
dhprebn  = dbndiff.clone()                                          # (32, 64)，第一部分
dbnmeani = (-dbndiff).sum(0, keepdim=True)                          # (1, 64)

# Step 6: bnmeani = hprebn.sum(0, keepdim=True) / n
# 前向 sum → 反向广播
dhprebn += (1.0 / n) * dbnmeani                       # ⚠️ += 累加第二部分
```

### 4.10 穿过第一个线性层

前向：`hprebn = embcat @ W1 + b1`

和 4.6 完全相同的套路：

```python
dembcat = dhprebn @ W1.T          # (32, 64) @ (64, 30) → (32, 30)
dW1     = embcat.T @ dhprebn      # (30, 32) @ (32, 64) → (30, 64)
db1     = dhprebn.sum(0)          # (32, 64) → (64,)  注意：无 keepdim，因为 b1 是一维
```

### 4.11 穿过 view 和 embedding 查表

**view 的反向**：view 只改变数据的逻辑形状，不改变内存。反向直接 view 回去。

```python
demb = dembcat.view(emb.shape)    # (32, 30) → (32, 3, 10)
```

**embedding 查表的反向**：前向是按索引从 `C` 中抽取行，反向就是把梯度按原路送回去。同一行被用多次 → 累加。

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
```

**至此，Exercise 1 完成！** 从 loss 一路手写反向传播到 embedding 查找表。

---

## 5. Exercise 2：cross-entropy 解析反向

### 动机

Exercise 1 把 cross-entropy 拆成了很多步逐一反向。但直接对数学表达式求导，很多项会约掉化简，得到一个极其简洁的公式。

### 数学推导

对单个样本，$L = -\log \frac{e^{l_y}}{\sum_j e^{l_j}}$，求 $\frac{\partial L}{\partial l_i}$：

- 当 $i \neq y$（非正确类别）：$\frac{\partial L}{\partial l_i} = P_i$（softmax 概率）
- 当 $i = y$（正确类别）：$\frac{\partial L}{\partial l_i} = P_i - 1$

一句话：**dlogits = softmax概率，正确类别位置减 1，再除以 n。**

### 代码

```python
dlogits = F.softmax(logits, dim=1)
dlogits[range(n), Yb] -= 1
dlogits /= n
```

三行代码替代了 Exercise 1 中 5 个子节的工作。

### 直觉：推拉力模型

把梯度想象成"力"：
- 正梯度 → 推下去（降低错误类别的 logit）
- 负梯度 → 拉上来（提高正确类别的 logit）
- **每行梯度之和 = 0**：推和拉的力完全平衡
- **力的大小 ∝ 预测概率**：模型越自信地犯错，受到的纠正力越大
- 如果预测完全正确（正确类别概率 = 1），则梯度全为 0，不需要调整

---

## 6. Exercise 3：BatchNorm 解析反向

### 动机

和 Exercise 2 同理——把 BatchNorm 标准化视为一个整体，直接推导解析公式。

### 数学设定

$$\mu = \frac{1}{n}\sum_i x_i, \quad \sigma^2 = \frac{1}{n-1}\sum_i (x_i - \mu)^2, \quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

已知 $\frac{\partial L}{\partial \hat{x}_i}$，求 $\frac{\partial L}{\partial x_i}$。

### 关键推导步骤

1. **$\frac{\partial L}{\partial \hat{x}_i}$**：从 $y_i = \gamma \hat{x}_i + \beta$ 直接得到，= $\frac{\partial L}{\partial y_i} \cdot \gamma$
2. **$\frac{\partial L}{\partial \sigma^2}$**：$\sigma^2$ 是标量，有 32 条路径，要对所有 $i$ 求和
3. **$\frac{\partial L}{\partial \mu}$**：有 33 条路径（32 条通过 $\hat{x}_i$ + 1 条通过 $\sigma^2$）。但 $\frac{\partial \sigma^2}{\partial \mu}$ 在 $\mu = \frac{1}{n}\sum x_i$ 的条件下恰好 = 0，整条路消失
4. **$\frac{\partial L}{\partial x_i}$**：三条路贡献汇总，展开化简

### 推导陷阱

代入表达式时注意**求和变量的命名冲突**：$\frac{\partial L}{\partial \sigma^2}$ 内部用 $\sum_i$，代入 $\frac{\partial L}{\partial x_i}$ 时外层也有 $i$，必须把内层改名为 $j$。

### 最终代码

```python
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact
    - dhpreact.sum(0)
    - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0)
)
```

一行代码同时对 64 个神经元并行执行 BatchNorm 反向传播。

### Bessel's Correction 补充

Andrej 使用 `1/(n-1)` 而非 `1/n` 计算方差：
- `1/n` 是有偏估计，系统性低估方差
- `1/(n-1)` 是无偏估计（Bessel's correction）
- 原始 BatchNorm 论文训练时用 `1/n`，推理时用 `1/(n-1)`——造成 train/test 不一致
- PyTorch 的 `BatchNorm1d` 继承了这个不一致
- Andrej 认为这是一个 bug，选择一律用 `1/(n-1)`

---

## 7. Exercise 4：全部整合，替代 loss.backward()

把 Exercise 2（cross-entropy）、Exercise 3（BatchNorm）和 Exercise 1 中其他层的代码拼在一起，用 `torch.no_grad()` 包裹训练循环，完全替代 `loss.backward()`。

```python
with torch.no_grad():
    # forward pass ...
    
    # backward pass (手写)
    dlogits = F.softmax(logits, dim=1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    
    dhpreact = (1.0 - h**2) * dh
    
    dhprebn = bngain * bnvar_inv / n * (
        n * dhpreact - dhpreact.sum(0)
        - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)
    )
    
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    
    demb = dembcat.view(emb.shape)
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
        for j in range(Xb.shape[1]):
            ix = Xb[k, j]
            dC[ix] += demb[k, j]
    
    # parameter update
    for p, grad in zip(parameters, grads):
        p.data += -lr * grad
```

结果：loss 和用 `loss.backward()` 的版本基本相同，采样输出质量一致。**大约 20 行代码就是整个网络的反向传播。**

---

## 8. 全课总结

> *"Hopefully you're looking at the backward pass of this neural net and you're thinking to yourself: actually, that's not too complicated."* — Andrej Karpathy

这节课的收获：

1. **直觉**：知道梯度如何从 loss 流过每一个变量
2. **调试能力**：如果梯度出了问题，你知道去哪里找
3. **理解 PyTorch 在背后做什么**：`loss.backward()` 不再是黑箱
4. **解析公式的力量**：cross-entropy 和 BatchNorm 的解析反向比逐步拆解快得多

每一层的反向传播大概就 2-3 行代码，**除了 BatchNorm 的解析公式稍微复杂之外，其他都很直接。**

下一课预告：进入 RNN 和 LSTM，开始让架构变得更复杂。

---

## 附录 A：反向传播规则速查表

| 规则 | 前向操作 | 反向操作 | 示例 |
|------|---------|---------|------|
| **链式法则** | `c = f(a)` | `da = f'(a) * dc` | 所有操作的基础 |
| **逐元素乘法** | `c = a * b` | `da = b * dc`, `db = a * dc` | BN gain/raw |
| **矩阵乘法** | `D = A @ B` | `dA = dD @ B.T`, `dB = A.T @ dD` | 线性层 |
| **加法 / bias** | `c = a + b` | `da = dc`, `db = dc` | bias 项 |
| **广播反向** | `b` 被复制到多行/列 | `db = dc.sum(广播维度)` | bngain, bnbias, b2 |
| **sum 反向** | `b = a.sum(dim)` | `da = db` 广播回去 | bnvar, bnmeani |
| **变量多次使用** | `a` 出现在多处 | 各路梯度 `+=` 累加 | bndiff, hprebn, counts |
| **log** | `b = log(a)` | `da = (1/a) * db` | logprobs |
| **exp** | `b = exp(a)` | `da = b * db`（用输出） | counts |
| **tanh** | `b = tanh(a)` | `da = (1 - b²) * db`（用输出） | h |
| **幂函数** | `b = a^n` | `da = n * a^(n-1) * db` | bnvar_inv |
| **max** | `b = max(a)` | 只有 argmax 位置有梯度 = 1 | logit_maxes |
| **view / reshape** | `b = a.view(shape)` | `da = db.view(a.shape)` | embcat → emb |
| **索引查表** | `b = table[idx]` | 按索引 `+=` 回 table | embedding |

**Shape 匹配法**：不记公式，看 `d变量` 的 shape 必须 = 变量的 shape，然后凑矩阵乘法的维度，转置自然就出来了。

---

## 附录 B：完整反向传播代码

Exercise 1 的完整逐步反向传播（未使用解析简化）：

```python
# --- Cross-Entropy Loss 拆解反向 ---
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0 / n

dprobs = (1.0 / probs) * dlogprobs

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts = counts_sum_inv * dprobs

dcounts_sum = (-counts_sum**(-2)) * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum

dnorm_logits = counts * dcounts

dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]).float() * dlogit_maxes

# --- 第二个线性层 ---
dh  = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

# --- tanh ---
dhpreact = (1.0 - h**2) * dh

# --- BatchNorm scale & shift ---
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw  = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)

# --- BatchNorm 标准化内部 ---
dbndiff    = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

dbnvar   = (-0.5) * (bnvar + 1e-5)**(-1.5) * dbnvar_inv
dbndiff2 = (1.0 / (n - 1)) * dbnvar
dbndiff += 2 * bndiff * dbndiff2

dhprebn  = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0, keepdim=True)
dhprebn += (1.0 / n) * dbnmeani

# --- 第一个线性层 ---
dembcat = dhprebn @ W1.T
dW1     = embcat.T @ dhprebn
db1     = dhprebn.sum(0)

# --- view + embedding ---
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]
```
