# The spelled-out intro to language modeling: building makemore 完整笔记

> 整理自 Andrej Karpathy 《The spelled-out intro to language modeling: building makemore》课程，从零实现字符级语言模型（Bigram），涵盖统计法与神经网络两种实现方式，结合学习过程中的疑问补充说明，适合上传GitHub存档，便于后续回顾与复用。

# 课程概述

本课程核心目标：从零搭建一个字符级语言模型（以Bigram二元语法模型为核心），通过「统计计数法」和「单层神经网络法」两种方式实现名字生成，理解语言模型的底层逻辑——基于前一个字符预测下一个字符，掌握PyTorch张量操作、梯度下降、反向传播等基础知识点，为后续学习MLP、Transformer等复杂模型奠定基础。

补充说明：学习过程中重点解决了库的安装与环境配置、conda使用、张量归一化关键参数等疑问，相关内容已融入对应知识点，确保笔记兼具完整性与实用性。

# 一、前置准备（环境+基础概念）

## 1.1 环境配置（重点解决学习疑问）

课程需用到Python、PyTorch、matplotlib三个工具，结合之前疑问，详细说明环境搭建步骤（适配conda环境，避免版本冲突）：

### 1.1.1 核心概念答疑

- **库是什么？**：别人写好的现成工具代码包，无需从零编写核心功能（如torch用于张量计算，matplotlib用于可视化），直接import即可调用。

- **库安装在哪里？**：安装在当前激活的Python环境中，不同conda环境互不干扰，避免库版本冲突（适合同时学习多个项目/课程）。

- **pip是什么？**：Python的“应用商店”，专门用于下载、安装、更新Python库，指令简单且通用。

- **conda是什么？**：AI专用的环境管家+万能安装器，既能创建独立Python环境，也能解决复杂库的依赖冲突，比pip更适合深度学习场景（推荐使用Miniconda轻量版）。

### 1.1.2 具体环境搭建步骤

1. 安装Miniconda：官网（https://docs.conda.io/en/latest/miniconda.html）下载对应系统版本，默认安装即可。

2. 打开终端/Anaconda Prompt（Windows优先用Anaconda Prompt），创建专属环境：
        `conda create -n makemore python=3.10  # 环境名makemore，Python版本3.10（适配PyTorch）
conda activate makemore  # 激活环境，成功标志：终端前显示(makemore)`

3. 安装所需库：`pip install torch  # CPU版足够本课程使用，无需GPU
pip install matplotlib  # 用于可视化计数矩阵`

### 1.1.3 Conda常用命令（必备）

```bash
conda env list  # 查看电脑中所有conda环境
conda deactivate  # 退出当前环境
conda remove -n makemore --all  # 删除makemore环境（无需时使用）
pip list  # 查看当前环境已安装的库
```

## 1.2 课程核心基础概念

- **Bigram（二元语法模型）**：最简单的字符级语言模型，仅根据「前一个字符」预测「下一个字符」，核心是学习字符对（ch1, ch2）的出现概率。

- **张量（Tensor）**：PyTorch中的核心数据结构，可理解为“多维数组”，本课程主要用2维张量（27×27）存储字符对的出现次数。

- **独热编码（One-Hot Encoding）**：将整数索引转为固定维度的向量（本课程为27维），解决神经网络无法直接处理整数的问题，特点是仅对应索引位为1，其余为0。

- **损失函数（负对数似然NLL）**：衡量模型预测效果的指标，损失越小，模型对训练数据的拟合度越好，核心是惩罚预测概率低的真实字符对。

- **反向传播**：PyTorch自动计算梯度的核心机制，从输出损失反向遍历计算图，得到所有参数的梯度，用于后续参数更新。

# 二、核心代码实现（完整可运行）

本课程分为两大模块：统计法实现Bigram模型、神经网络法实现Bigram模型，代码可直接复制运行（需确保已激活makemore环境，且准备好names.txt数据集）。

## 2.1 数据准备与字符映射

核心任务：读取人名数据集，构建字符与整数索引的映射（计算机仅能处理数字，无法直接识别字母），添加特殊符号「.」作为名字的起始/结束标记。

```python
import torch
import matplotlib.pyplot as plt

# 1. 读取人名数据集（names.txt为课程配套数据集，需与代码同目录）
words = open('names.txt', 'r').read().splitlines()
print(f"数据集共包含 {len(words)} 个英文名字")
print(f"前10个名字：{words[:10]}")

# 2. 构建字符词汇表与索引映射
# 收集所有不重复字符并排序
chars = sorted(list(set(''.join(words))))
# 字符→整数（stoi: string to index），预留0给特殊符号.
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0  # 特殊符号.对应索引0，代表名字的开始和结束
# 整数→字符（itos: index to string），用于后续生成名字时转换
itos = {i: ch for ch, i in stoi.items()}
print(f"\n字符索引映射：{itos}")
print(f"共包含 {len(itos)} 个符号（26个字母+1个特殊符号.）")
```

## 2.2 统计法实现Bigram模型

核心逻辑：统计所有字符对（ch1, ch2）的出现次数，归一化为概率矩阵，基于概率采样生成新名字，评估模型损失。

### 2.2.1 统计字符对出现次数

```python
# 初始化27×27的整数张量，存储字符对出现次数（行=当前字符，列=下一个字符）
N = torch.zeros((27, 27), dtype=torch.int32)

# 遍历每个名字，统计所有字符对
for word in words:
    # 给每个名字添加起始和结束标记.
    chs = ['.'] + list(word) + ['.']
    # 遍历连续的字符对（ch1=前一个字符，ch2=后一个字符）
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]  # 转换为整数索引
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1  # 对应位置计数+1

print(f"\n字符对计数矩阵（前5行5列）：\n{N[:5, :5]}")
```

### 2.2.2 可视化计数矩阵

```python
# 创建画布，设置大小（16×16更清晰）
plt.figure(figsize=(16, 16))
# 绘制热力图，蓝色越深代表出现次数越多
plt.imshow(N, cmap='Blues')

# 遍历每个格子，标注字符对和计数
for i in range(27):
    for j in range(27):
        # 标注字符对（如.e、em）
        ch_str = itos[i] + itos[j]
        plt.text(j, i, ch_str, ha="center", va="center", color="gray")
        # 标注计数数字
        plt.text(j, i+0.3, N[i, j].item(), ha="center", va="center", color="black")

# 隐藏坐标轴，提升美观度
plt.axis('off')
plt.show()  # 显示热力图，可直观看到常见字符对
```

可视化说明：第一行（索引0，对应.）代表名字的开头，可看到a、e等字母列的计数较高，说明很多名字以a、e开头；最后一列（索引0，对应.）代表名字的结尾，可看到多数字符最终指向.。

### 2.2.3 计数转概率（归一化，重点解答keepdim=True疑问）

```python
# 将整数计数转为浮点数，便于计算概率
P = N.float()

# 按行归一化，让每一行的概率总和为1（核心：keepdim=True）
# 重点答疑：为什么必须加keepdim=True？
# 不加keepdim=True：P.sum(1)输出形状为[27]（一维），除法会按列广播，导致计算错误
# 加keepdim=True：P.sum(1, keepdim=True)输出形状为[27, 1]（二维），可精准对每一行单独归一化
P /= P.sum(1, keepdim=True)

print(f"\n概率矩阵（前5行5列）：\n{P[:5, :5]}")
print(f"每一行概率和：{P.sum(1)[:5]}")  # 验证每行和为1
```

### 2.2.4 基于概率采样生成名字

```python
# 固定随机种子，保证每次生成结果可复现
g = torch.Generator().manual_seed(2147483647)

# 生成10个名字
print("\n统计法生成的10个名字：")
for i in range(10):
    out = []
    ix = 0  # 起始字符为.（索引0）
    while True:
        p = P[ix]  # 获取当前字符的概率分布
        # 按概率采样下一个字符的索引（multinomial：按概率采样）
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])  # 转换为字符，加入结果列表
        if ix == 0:  # 遇到.，代表名字结束，退出循环
            break
    print(''.join(out))
```

### 2.2.5 损失函数计算（负对数似然NLL）

```python
# 重点答疑：为什么log(prob)都是负数？
# 因为概率prob的取值范围是0~1，而对数函数ln(x)在0<x<1时，结果为负数；prob越接近1，log(prob)越接近0，总和越大，模型越准

log_likelihood = 0.0  # 总对数似然（初始为0）
n = 0  # 总样本数（所有字符对的数量）

for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]  # 模型预测的概率
        logprob = torch.log(prob)  # 取对数
        log_likelihood += logprob  # 累加所有样本的对数概率
        n += 1

# 负对数似然（NLL）：加负号转为最小化目标，取平均值使结果更具参考性
nll = -log_likelihood / n
print(f"\n统计法模型损失（NLL）：{nll.item():.4f}")
```

### 2.2.6 模型平滑（解决零概率问题）

```python
# 问题：部分字符对从未出现过，概率为0，log(0)会得到-∞，导致损失爆炸
# 解决方法：加一平滑（给所有计数加1，让所有字符对都有微小概率）
P_smoothed = (N + 1).float()  # 所有计数+1
P_smoothed /= P_smoothed.sum(1, keepdim=True)  # 重新归一化

# 重新计算平滑后的损失
log_likelihood_smoothed = 0.0
for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P_smoothed[ix1, ix2]
        log_likelihood_smoothed += torch.log(prob)

nll_smoothed = -log_likelihood_smoothed / n
print(f"平滑后模型损失（NLL）：{nll_smoothed.item():.4f}")  # 损失会降低，模型更稳定
```

## 2.3 神经网络法实现Bigram模型

核心逻辑：用单层线性神经网络复现统计法模型，理解「统计计数 ≡ 单层线性网络 + Softmax」的等价关系，掌握神经网络的前向传播、反向传播与参数更新。

### 2.3.1 构建训练数据集（x=输入字符，y=目标字符）

```python
# xs：输入（前一个字符的索引），ys：目标（下一个字符的索引）
xs, ys = [], []

for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# 转换为PyTorch张量
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()  # 总样本数
print(f"\n训练集总样本数：{num}")
print(f"输入xs前10个：{xs[:10]}")
print(f"目标ys前10个：{ys[:10]}")
```

### 2.3.2 独热编码（输入预处理，解答独热编码作用）

```python
import torch.nn.functional as F

# 重点答疑：为什么需要独热编码？
# 神经网络无法直接处理整数索引（整数有大小关系，而字符无大小关系）
# 独热编码将整数转为27维向量，仅对应索引位为1，其余为0，消除大小关系影响
xenc = F.one_hot(xs, num_classes=27).float()  # 转为27维独热向量，浮点数类型（适配神经网络）

print(f"\n独热编码后形状：{xenc.shape}")  # 形状：(总样本数, 27)
print(f"第一个输入的独热编码：{xenc[0]}")  # 对应xs[0]的索引，仅对应位置为1
```

### 2.3.3 定义单层线性神经网络

```python
# 初始化权重矩阵W（27输入×27输出，对应27个字符）
# requires_grad=True：告诉PyTorch需要计算该参数的梯度，用于后续反向传播
W = torch.randn((27, 27), requires_grad=True)

# 前向传播（核心：线性计算 + Softmax转概率）
# 1. 线性计算：logits = 独热编码 × 权重矩阵（未归一化的得分）
logits = xenc @ W
# 2. 指数化：将logits转为正数，对应统计法中的“计数”
counts = logits.exp()
# 3. 归一化：转为概率分布（与统计法的P矩阵等价）
probs = counts / counts.sum(1, keepdim=True)

# 计算神经网络初始损失
loss = -probs[torch.arange(num), ys].log().mean()
print(f"\n神经网络初始损失：{loss.item():.4f}")
```

### 2.3.4 反向传播与参数更新（梯度下降）

```python
# 学习率：控制参数更新步长（太大易震荡，太小训练慢，本课程设50合适）
lr = 50
# 训练轮次：100轮，足够损失收敛到接近统计法的结果
epochs = 100

print("\n神经网络训练过程（每10轮打印一次损失）：")
for k in range(epochs):
    # 1. 前向传播（重复计算，确保每轮都基于最新权重）
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    
    # 2. 计算损失（加入L2正则化，等价于统计法的加一平滑，防止过拟合）
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    
    # 3. 反向传播：自动计算梯度
    W.grad = None  # 重置梯度（必须做！否则梯度会累加，导致更新错误）
    loss.backward()  # 自动计算W的梯度
    
    # 4. 梯度下降更新权重：W = W - 学习率 × 梯度
    W.data += -lr * W.grad
    
    # 每10轮打印一次损失，查看训练效果
    if k % 10 == 0:
        print(f"轮次 {k:3d} | 损失: {loss.item():.4f}")

print(f"\n神经网络训练完成，最终损失：{loss.item():.4f}")  # 最终损失与统计法接近
```

### 2.3.5 用训练好的神经网络生成名字

```python
# 固定随机种子，与统计法一致，对比生成效果
g = torch.Generator().manual_seed(2147483647)

print("\n神经网络生成的10个名字：")
for i in range(10):
    out = []
    ix = 0  # 起始字符为.（索引0）
    while True:
        # 输入字符转独热编码
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        # 前向传播计算概率
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        # 按概率采样下一个字符
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:  # 遇到.结束
            break
    print(''.join(out))  # 生成结果与统计法几乎一致，验证等价性
```

# 三、高频疑问汇总（整合学习过程中的问题）

## 3.1 环境相关

- Q：库是什么？装在哪里？如何安装？
A：库是现成的工具代码包，装在当前激活的Python环境中；通过pip install 库名安装，结合conda环境可避免版本冲突。

- Q：conda是什么？如何创建、激活环境？
A：conda是环境管家，用于创建独立Python环境；创建指令conda create -n 环境名 python=版本，激活指令conda activate 环境名。

## 3.2 代码与原理相关

- Q：归一化时，keepdim=True为什么必须加？
A：不加会导致求和结果为一维，除法按列广播，计算错误；加了保持二维形状，可精准对每一行单独归一化（每行概率和为1）。

- Q：log(prob)为什么都是负数？
A：因为prob∈(0,1)，对数函数ln(x)在0<x<1时结果为负；prob越接近1，log(prob)越接近0，总和越大，模型越准。

- Q：为什么需要独热编码？
A：神经网络无法处理整数索引（整数有大小关系，字符无），独热编码将整数转为无大小关系的向量，适配神经网络输入。

- Q：模型平滑的作用是什么？
A：解决部分字符对概率为0的问题，避免log(0)导致损失爆炸，同时提升模型泛化能力。

- Q：神经网络与统计法为什么生成结果一致？
A：两者本质等价：统计法的计数矩阵对应神经网络的logits.exp()，概率矩阵对应Softmax后的结果，训练后权重W拟合出统计概率。

# 四、课程核心总结

## 4.1 核心逻辑链

数据准备（读取名字+字符映射）→ 统计法（计数→归一化→采样→损失）→ 神经网络法（独热编码→线性层→Softmax→反向传播→参数更新）→ 验证两者等价性。

## 4.2 关键收获

1. Bigram模型核心：仅基于前一个字符预测下一个字符，是最简单的字符级语言模型。

2. 统计法与神经网络法等价：单层线性神经网络+Softmax，本质是在学习字符对的统计概率。

3. 深度学习底层逻辑：自动微分（反向传播）是核心，PyTorch通过requires_grad=True实现梯度自动计算。

4. 损失函数意义：负对数似然（NLL）衡量模型预测精度，损失越小，模型越准；平滑处理可避免模型崩溃。

## 4.3 延伸方向

- 升级模型：Bigram仅看前一个字符，效果有限，后续可学习MLP、RNN、Transformer等复杂模型，引入更长上下文。

- 优化策略：可补充其他激活函数（ReLU）、损失函数（交叉熵）、批量训练、学习率调度等。

- 工程扩展：结合PyTorch的GPU加速、批量运算，提升模型训练效率。

# 五、GitHub上传补充说明

1. 文件命名：建议命名为「makemore_course_notes.md」，便于识别和检索。

2. 文件结构：可将笔记与names.txt数据集、单独的代码文件（makemore_demo.py）放在同一仓库，代码文件可提取笔记中的核心代码，便于直接运行。

3. 渲染说明：GitHub原生支持LaTeX公式和Python代码高亮，笔记中的公式、代码块可正常显示，无需额外配置。

4. 运行环境：在仓库README中补充环境搭建步骤（可直接复制本笔记的1.1.2节），方便他人复现。
> （注：文档部分内容可能由 AI 生成）