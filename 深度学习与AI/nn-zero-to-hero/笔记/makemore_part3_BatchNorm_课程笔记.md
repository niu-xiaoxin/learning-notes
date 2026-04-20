# Building makemore Part 3: Activations, Gradients & BatchNorm 学习笔记

> 本笔记基于 Andrej Karpathy 的 "Building makemore Part 3" 课程整理，记录了神经网络训练中激活值、梯度的诊断与 Batch Normalization 的核心原理。

---

## 目录

- [0. 课程背景与动机](#0-课程背景与动机)
- [1. 初始化问题之一：Softmax 层的过度自信](#1-初始化问题之一softmax-层的过度自信)
- [2. 初始化问题之二：Tanh 层的饱和问题](#2-初始化问题之二tanh-层的饱和问题)
- [3. Kaiming 初始化：有原则地设定权重缩放](#3-kaiming-初始化有原则地设定权重缩放)
- [4. Batch Normalization 的核心思想](#4-batch-normalization-的核心思想)
- [5. BatchNorm 的副作用：样本耦合与推理问题](#5-batchnorm-的副作用样本耦合与推理问题)
- [6. BatchNorm 的两个重要细节](#6-batchnorm-的两个重要细节)
- [7. BatchNorm 总结与代码 PyTorch 化](#7-batchnorm-总结与代码-pytorch-化)
- [8. 诊断工具之一：前向激活值统计](#8-诊断工具之一前向激活值统计)
- [9. 诊断工具之二：反向梯度统计](#9-诊断工具之二反向梯度统计)
- [10. 诊断工具之三:参数与 Update-to-Data 比率](#10-诊断工具之三参数与-update-to-data-比率)
- [11. BatchNorm 如何让一切变得更容易](#11-batchnorm-如何让一切变得更容易)
- [12. 最终总结与反思](#12-最终总结与反思)
- [附录：个人思考与问答记录](#附录个人思考与问答记录)

---

## 0. 课程背景与动机

本课程的核心目标是**深入理解神经网络训练过程中激活值（activations）和梯度（gradients）的行为**。

### 为什么重要？

RNN 虽然理论上是通用近似器，但实践中很难用一阶梯度下降优化。要理解"为什么难优化"，关键就在于理解激活值和梯度的动态。后续 RNN 的各种改进变体（GRU、LSTM 等）本质上都是在解决这个问题。

### 本课要达成的目标

1. 建立对激活值和梯度的直觉理解
2. 学会神经网络的正确初始化方式
3. 学习 Batch Normalization —— 里程碑式的技术创新
4. 掌握诊断工具,判断网络训练状态是否健康

---

## 1. 初始化问题之一：Softmax 层的过度自信

### 问题现象

训练开始时观察到第 0 次迭代的 loss 高达 **27**,然后迅速降到 2 左右。这是一个严重的红旗。

### 理论预期值

对于 27 个字符(26 字母 + 结束符),初始化时网络应该输出均匀分布,每个字符概率为 1/27:

```
expected_loss = -log(1/27) ≈ 3.29
```

### 问题根源

初始化时 logits 取了极端值,经过 softmax 产生了**非常尖锐的概率分布**——网络**自信地给出了错误答案**,所以 loss 才会这么高。

### 简单的例子

- logits 全为 0 → softmax 得到均匀分布 → loss = -log(0.25) = 1.38 ✅
- logits 是极端随机值 → softmax 某个类别概率接近 1 → 大概率猜错 → loss 爆炸

### 解决方案

logits 由 `h @ W2 + b2` 计算得到,要让 logits 接近零:

```python
# 将偏置初始化为 0
b2 = torch.randn(vocab_size) * 0  # 或直接设为 0

# 缩小权重到很小的值
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
```

### 为什么不把 W2 设为精确的零?

这会破坏**对称性破缺(symmetry breaking)**。如果所有权重完全相同,所有神经元会做相同的事,失去多个神经元的意义。用很小但非零的值(如 0.01)既能让 loss 接近理想值,又保留随机性。

### 修复后的效果

- 初始 loss 从 27 降到约 3.3 ✅
- loss 曲线不再有"冰球棍"形状(开头急剧下降只是在浪费训练时间)
- 验证集 loss 从 2.17 改善到 **2.13**

---

## 2. 初始化问题之二：Tanh 层的饱和问题

### 问题现象

查看隐藏层激活值 `h = tanh(h_preact)` 的直方图,发现**绝大多数值集中在 -1 和 +1**。

### 为什么这是问题?

tanh 的反向传播公式:`local_gradient = 1 - t²`

- `t` 接近 0 → `1 - t² ≈ 1`,梯度完整通过 ✅
- `t` 接近 ±1 → `1 - t² ≈ 0`,**梯度被杀死** ❌

直觉上:当 tanh 输出在平坦区域,无论怎么改变输入,输出几乎不变,对 loss 没影响,梯度自然为零。

### 死神经元(Dead Neurons)

- **可视化方法**:画 32×200 的布尔矩阵,白色表示 `|h| > 0.99` 的饱和状态
- **最糟情况**:某一列完全是白色 → 该神经元对所有样本都饱和 → **永远学不到东西**
- 不仅 tanh,sigmoid、ReLU、ELU 等激活函数都有类似问题(ReLU 死亡尤其致命,称为"永久性脑损伤")

### 解决方案

同样是缩小预激活值:

```python
W1 *= 0.2     # 大幅缩小权重
b1 *= 0.01    # 小一点的偏置,保留随机性
```

修复后预激活值范围从 [-15, +15] 缩小到 [-1.5, +1.5]。

### 修复后的效果

- tanh 输出不再挤在 ±1,饱和率大幅降低
- 验证集 loss 从 2.13 进一步改善到 **2.10**

### 剩下的问题

`0.2` 这个数字是手工试出来的。如果网络有很多层,不可能手工调每一层。需要**有原则的方法**。

---

## 3. Kaiming 初始化：有原则地设定权重缩放

### 核心问题

矩阵乘法会改变分布的宽度:

```python
x = torch.randn(1000, 10)   # 标准差 = 1
w = torch.randn(10, 200)
y = x @ w                    # 标准差 ≈ 3
```

### 数学结论

要保持输出标准差为 1,需要**除以 √fan_in**(fan_in 是输入维度)。

**直觉**:矩阵乘法输出的每个元素是 fan_in 个乘积的和。方差相加变成 fan_in,标准差变成 √fan_in。除以 √fan_in 正好抵消这个放大。

### Kaiming He 论文的结论

不同激活函数需要不同的 gain:

| 激活函数 | Gain | 权重标准差 |
|---------|------|-----------|
| Linear/无激活 | 1 | 1/√fan_in |
| ReLU | √2 | √(2/fan_in) |
| Tanh | 5/3 | (5/3)/√fan_in |

**为什么 ReLU 是 √2?** 因为 ReLU 砍掉一半负值分布,需要乘 √2 补偿。

**为什么 Tanh 是 5/3?** Karpathy 坦言他也不完全确定出处,但经验上这个值对 "Linear + Tanh 三明治"效果最好。

### PyTorch 实现

```python
torch.nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
```

### 应用到我们的网络

W1 的 fan_in = 10(嵌入维度) × 3(block_size) = 30

```python
std = (5/3) / sqrt(30) ≈ 0.3
W1 = torch.randn(...) * 0.3
```

之前手调是 0.2,有原则地算出来是 0.3,训练结果都是 loss = 2.10。

### Karpathy 的重要观点

精确初始化在 7 年前极其关键,但现代创新让它不再那么重要:

1. **残差连接(Residual Connections)**
2. **归一化层**(BatchNorm、LayerNorm、GroupNorm 等)
3. **更好的优化器**(Adam、RMSProp)

---

## 4. Batch Normalization 的核心思想

### 核心洞见

既然我们希望隐藏状态服从单位高斯分布,**那为什么不直接归一化它们?**

关键前提:**"把张量标准化为单位高斯"是完全可微的操作**,可以放在网络里正常梯度传播。

### 具体实现

```python
# 沿 batch 维度(第 0 维)计算
bnmean = h_preact.mean(dim=0, keepdim=True)   # 形状 1×200
bnstd = h_preact.std(dim=0, keepdim=True)     # 形状 1×200

h_preact_normalized = (h_preact - bnmean) / bnstd
```

注意:沿 **batch 维度**取统计量——对每个神经元,在 32 个样本上归一化。这就是 **Batch** Normalization 的含义。

### 加上 scale 和 shift

但强制每层永远是单位高斯过于僵硬。加入可学习参数让网络自己决定分布:

```python
bngain = torch.ones(1, 200)   # gamma,初始化为 1
bnbias = torch.zeros(1, 200)  # beta,初始化为 0

out = bngain * h_preact_normalized + bnbias
```

- **初始化时**:gamma=1, bias=0 → 输出是单位高斯 ✅
- **训练中**:通过反向传播自由调整分布形状

### 在网络中的位置

标准模式:

```
Linear → BatchNorm → Activation → Linear → BatchNorm → Activation → ...
```

---

## 5. BatchNorm 的副作用：样本耦合与推理问题

### 副作用:样本之间被耦合

**BatchNorm 之前**:每个样本独立处理,输出只取决于自身输入。

**BatchNorm 之后**:一个样本的输出依赖于**同 batch 中其他样本**——均值和标准差是全 batch 共同决定的。

### Jitter 效应

同一个样本 "emma" 在不同 batch 中会得到**略有不同**的激活值。

### 出乎意料:这居然是好事

这种抖动起到了**正则化**作用——类似数据增强,让网络更难过拟合到具体样本。

### 但没人喜欢这个性质

- 导致大量难以调试的 bug
- Batch size 改变会影响结果
- 破坏了"每个样本独立处理"的优雅性

替代方案:**Layer Normalization**、Instance Normalization、Group Normalization。

### 推理时的问题

训练时用 batch 统计量,但**部署时用户一次只发一个样本**,均值为它自己、标准差为 0,公式崩溃。

### 解决方案一:训练后单独校准

```python
with torch.no_grad():
    # 跑一遍整个训练集
    bnmean = hpreact.mean(0, keepdim=True)
    bnstd = hpreact.std(0, keepdim=True)
```

Karpathy 吐槽:"没人喜欢搞第二阶段,大家都懒。"

### 解决方案二:训练中维护 running average

```python
bnmean_running = torch.zeros(1, n_hidden)
bnstd_running = torch.ones(1, n_hidden)

# 每次训练迭代
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmean_i
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstd_i
```

### Parameter vs Buffer

- **Parameters**(通过反向传播学习):gamma, beta
- **Buffers**(运行平均更新,不走梯度):running_mean, running_std

PyTorch 官方实现用的是方案二。

---

## 6. BatchNorm 的两个重要细节

### 细节一:Epsilon 的作用

标准化公式实际是:

```
x_hat = (x - mean) / √(variance + ε)
```

**作用**:防止方差为 0 时除零(极端情况下某神经元对全 batch 输出相同值)。通常 ε = 1e-5,生产代码中务必添加。

### 细节二:前置线性层的 bias 是冗余的

**观察**:BatchNorm 会减去 batch 均值,而 b1 是加给所有样本的**同一个常数**——求均值时完整保留,标准化时被**完整减掉**。

```
(h_preact + b1) - mean(h_preact + b1) = h_preact - mean(h_preact)
# b1 完全消失!
```

**后果**:
- b1 的梯度永远为 0
- b1 永远不更新
- 白白浪费参数

### 正确模式

```python
# 错误:b1 冗余
h_preact = embcat @ W1 + b1

# 正确:BatchNorm 前省略 bias
h_preact = embcat @ W1
```

在 PyTorch 中:`nn.Linear(in, out, bias=False)`。

### 真实世界:ResNet 的代码

```python
self.conv1 = nn.Conv2d(..., bias=False)   # 注意!
self.bn1 = nn.BatchNorm2d(...)
self.relu = nn.ReLU()
```

这个 `Linear/Conv → Normalization → Nonlinearity` 的三明治是深度学习的基本构建块。

---

## 7. BatchNorm 总结与代码 PyTorch 化

### BatchNorm 层的完整结构

| 组件 | 类型 | 更新方式 |
|------|------|---------|
| gamma | Parameter | 反向传播 |
| beta | Parameter | 反向传播 |
| running_mean | Buffer | 指数移动平均 |
| running_var | Buffer | 指数移动平均 |

### PyTorch 化的模块代码

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Buffers
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
```

### 像乐高一样搭网络

```python
layers = [
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden, bias=False),            BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
]
```

### 前向传播

```python
x = emb.view(emb.shape[0], -1)
for layer in layers:
    x = layer(x)
loss = F.cross_entropy(x, Yb)
```

**关键**:这个 API 和 PyTorch 官方几乎一模一样,加 `nn.` 前缀就能直接跑。

---

## 8. 诊断工具之一：前向激活值统计

### 要观察什么

对每个 tanh 层的输出,计算:

1. **Mean** — 应接近 0
2. **Std** — 衡量分布宽度
3. **Saturation %** — `|t| > 0.97` 的比例(落在平坦尾部)

画所有层的直方图叠加对比。

### 理想状态(Gain = 5/3)

- 第一层饱和率约 20%(稍高但可接受)
- 后续层稳定在 **标准差约 0.65,饱和率约 5%**
- 所有层直方图**形状高度相似** ✅

### Gain 太小(比如 1)

- 每层分布逐渐变窄
- 深层激活值**塌缩到零**
- 信号在网络中消失

### Gain 太大(比如 3)

- 饱和率飙升
- 大量神经元落在 tanh 平坦区
- 梯度几乎无法流过

### 完全去掉 Tanh 的对比实验

纯线性层堆叠时:
- Gain = 5/3 过大,激活值深层扩散
- Gain = 1 时前向/反向都健康

**但纯线性层有致命问题**:数学上等价于单层线性变换

```
W3 @ (W2 @ (W1 @ x)) = (W3 @ W2 @ W1) @ x = W_combined @ x
```

**非线性函数的存在意义**:打破"坍缩",让深度网络拥有逼近任意函数的能力。

### 健康的标志

所有层激活值分布**形状相似、尺度一致(homogeneous)**。

---

## 9. 诊断工具之二：反向梯度统计

### 要观察什么

同样的方法,但把 `layer.out` 换成 `layer.out.grad`。

### 为什么所有层梯度应该相似?

梯度下降更新:`W = W - lr × W.grad`

所有层共用同一学习率。梯度差几个数量级 = 不同层在用不同学习率训练 → 训练极其不平衡。

### 两个工具必须配合使用

- 单看前向不够:可能前向 OK 但反向出问题
- 单看反向不够:可能反向 OK 但前向塌缩

**只有前向和反向同时稳定一致,才说明初始化健康**。

### 健康的标志

所有层梯度分布**一致**,没有:
- 逐层放大(梯度爆炸)
- 逐层缩小(梯度消失)
- 不对称(深层和浅层行为不同)

---

## 10. 诊断工具之三：参数与 Update-to-Data 比率

Karpathy 认为**最重要**的诊断工具。

### Gradient-to-Data Ratio

```
W.grad.std() / W.data.std()
```

衡量梯度尺度相对于参数尺度。

**发现的问题**:大部分层约 1e-3,但**最后一层约 1e-2**——最后一层训练速度是其他层的 10 倍。

**原因**:第 1 节中我们故意把最后一层权重缩小 10 倍(解决 softmax 过自信)——导致权重数值异常小,但梯度尺度正常。

### 更好的指标:Update-to-Data Ratio

```python
(lr * W.grad).std() / W.data.std()
```

然后取 log10。

**经验法则**:log10 应在 **-3 附近**(每次更新约为参数值的千分之一)。

- log10 >> -3 → 更新太大,可能发散
- log10 << -3 → 更新太小,训练太慢
- log10 ≈ -3 → 节奏刚好 ✅

### 追踪随时间的变化

```python
ud = []  # update-to-data ratios over time

for i in range(max_steps):
    # ... forward, backward, update ...
    with torch.no_grad():
        ratios = [
            ((lr * p.grad).std() / p.data.std()).log10().item()
            for p in parameters if p.ndim == 2
        ]
        ud.append(ratios)
```

画成曲线图,理想情况下所有线在 -3 附近徘徊。

### 通过这个图能诊断什么

| 现象 | 可能的问题 |
|------|-----------|
| 所有线压在 -4 | 学习率太低 |
| 层之间差异巨大 | 初始化未校准(如忘记 Kaiming) |
| 某层初期偏高,逐步收敛 | 该层被人为缩小(如最后一层) |

### 这个工具的威力

它综合了**学习率、梯度大小、参数大小**三者的关系,直接回答:"我的网络在以合适的速度学习吗?"

---

## 11. BatchNorm 如何让一切变得更容易

### 核心演示

把 BatchNorm 加回网络,观察三大诊断图的变化。

### 加了 BatchNorm 之后

| 指标 | 结果 |
|------|------|
| 前向激活值 | 所有层标准差约 0.65,饱和率约 2%,形状一致 ✅ |
| 反向梯度 | 所有层分布一致 ✅ |
| 参数/梯度 | 健康 ✅ |
| Update-to-data | 约 1e-3,略偏上 ✅ |

### 关键实验:改变 Linear 层的 Gain

**把 gain 改成 0.2(小得离谱)**:
- **激活值完全不受影响** ✅(BatchNorm 强制归一化)
- 梯度也不受影响
- **但** update-to-data 变小

**把 gain 改得很大**:
- 激活值依然不受影响
- update-to-data 变小

**结论**:
- **前向传播 + 反向梯度** → 对 gain 几乎免疫 ✅
- **参数更新速度** → 仍受 gain 影响,可能需要调学习率

### 完全去掉 Kaiming 初始化

用纯 `torch.randn` 初始化权重,在 BatchNorm 之前这是灾难。加了 BatchNorm:

- 前向/反向 → 依然健康 ✅
- Update-to-data → 偏低,把学习率调大 10 倍即可修复

### 深刻含义

**BatchNorm 让网络训练从"平衡铅笔"变成"搭乐高"**。以前需要精确调校每个环节,现在只需关注学习率校准。

---

## 12. 最终总结与反思

### 达成了什么

#### 目标一:介绍 Batch Normalization
第一个让深层网络可以稳定训练的现代创新。

#### 目标二:代码 PyTorch 化
把代码封装成可堆叠的"乐高积木"模块。

#### 目标三:介绍诊断工具
- 前向激活值分布
- 反向梯度分布
- 参数/梯度分布
- **Update-to-data 比率**(最重要)

### 没达成什么

- **没突破之前的性能**:Loss 仍是 2.10,因为**瓶颈在架构**(上下文太短只用 3 个字符),不在优化
- **没完整讲反向传播数学**:还有很多直觉理解的工作要做

### 研究前沿的现状

> 不用感到沮丧——我们正在触及这个领域的前沿。初始化没被真正"解决",反向传播也没被真正"解决"。这些仍是活跃研究领域。

### 展望 RNN

RNN 本质上是**非常深的网络**(展开循环后时间步 = 网络深度)。所有我们讲的稳定性问题在 RNN 中会以更极端形式出现——这就是 LSTM、GRU 等变体诞生的原因。

### 整体脉络回顾

```
发现问题(loss=27) 
  → 修 softmax
  → 修 tanh 饱和
  → 原则化(Kaiming 初始化)
  → BatchNorm 核心思想
  → BatchNorm 的代价与细节
  → PyTorch 化
  → 三大诊断工具
  → BatchNorm 让一切可管理
```

**核心主题:深度学习的历史,就是让神经网络训练从脆弱变稳健的历史**。

### 实践建议

**训练新网络时**:
- 检查初始 loss 是否符合理论预期
- Kaiming 初始化作为默认起点
- BatchNorm 前的线性/卷积层 `bias=False`
- 堆叠"线性 → 归一化 → 非线性"基本积木

**诊断网络时**:
- 画三张图:激活值、梯度、update-to-data
- 确认各层分布一致
- Update-to-data 目标约 **10^-3**

**遇到问题时**:
- 训练慢 → 看 update-to-data 是否过低
- 训练发散 → 看 update-to-data 是否过高
- 网络不学 → 看激活值是否塌缩/饱和
- 性能上不去 → 考虑**架构瓶颈**而非优化问题

---

## 附录:个人思考与问答记录

### Q: 为什么希望预激活值 `h_preact` 服从均值 0、标准差 1 的高斯分布?

这个问题背后其实有三层含义。

#### 第一层:避免 tanh 饱和(最直接)

- `h_preact` 很大 → tanh 输出接近 ±1 → 梯度接近 0 → 神经元死亡
- **均值 0** 保证激活值分布在 tanh 活跃区域中心
- **标准差 1** 让大部分值落在 [-2, +2] 之间——正好是 tanh 从线性过渡到饱和的"有趣区域",既有非线性又不饱和

#### 第二层:让信号在深层网络中不爆炸、不消失

想象 50 层网络,每层都对输入做 `y = W @ x + b`:

- 每层放大 1.5 倍 → 50 层后放大 **6 亿倍** → 信号爆炸
- 每层缩小到 0.8 倍 → 50 层后缩小到 **0.000014 倍** → 信号消失

**唯一稳定的选择**:让每层输入输出保持相同尺度——均值 0、标准差 1 最自然。

反向传播同理:梯度要穿过很多层,不稳定就会梯度爆炸/消失。

#### 第三层:为什么偏偏是高斯?

- **中心极限定理**:矩阵乘法是多个独立项求和,自然趋向高斯——不是强加,是顺应
- **对称无偏**:均值 0 保证没有系统性偏差
- **有限的二阶矩**:标准差 1 给了明确的尺度

#### 用一个比喻理解

训练神经网络像在一条河上送信:

- 河道太窄(标准差太小)→ 船搁浅(梯度消失)
- 河道太急(标准差太大)→ 船翻掉(梯度爆炸/饱和)
- **均值 0、标准差 1** = 每段河道水流速度差不多,船能顺畅来回

#### 完整含义

> 我们希望每层激活值和梯度都保持稳定合理的尺度,既不爆炸也不消失,既不让激活函数饱和也不让它退化成线性。"均值 0、标准差 1 的高斯分布"是满足这些要求最简洁、最自然的描述。

---

## 参考资料

- **原课程视频**:Andrej Karpathy - Building makemore Part 3: Activations, Gradients, BatchNorm
- **Kaiming He 论文**:"Delving Deep into Rectifiers" (2015)
- **BatchNorm 论文**:"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
- **PyTorch 文档**:`torch.nn.Linear`, `torch.nn.BatchNorm1d`, `torch.nn.init.kaiming_normal_`
