# Karpathy micrograd 课程完整笔记（手动实现自动微分与神经网络）

# 课程概述

本笔记记录 Andrej Karpathy 的 micrograd 课程核心内容，从零手动实现「自动微分引擎」，逐步搭建「神经元」和「多层感知机（MLP）」，理解深度学习框架（PyTorch/TensorFlow）的底层逻辑。

核心目标：掌握「计算图」「反向传播」「梯度下降」的底层原理，能用极简代码实现神经网络的前向预测与反向更新。

# 一、核心基础概念（前置知识）

## 1.1 导数与偏导数

- **导数**：单变量函数的变化率，即 \( f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \)，描述单个输入对输出的影响。

- **偏导数**：多变量函数中，固定其他变量，单独求某一个变量的导数（如 \( \frac{\partial f}{\partial a} \)，描述变量a对输出f的影响）。

- **关键区别**：导数对应单输入，偏导数对应多输入，梯度是所有偏导数的向量集合。

## 1.2 数值导数 vs 解析导数

- **数值导数**：用近似公式 \( \frac{f(x+h) - f(x)}{h} \)（h取极小值，如1e-7）计算，直观但精度低、速度慢，仅用于理解原理。

- **解析导数**：通过数学公式直接求导（如 \( f(x)=3x^2+4x+5 \)，导数 \( f'(x)=6x+4 \)），精确、高效，是神经网络的核心求导方式。

## 1.3 梯度与反向传播

- **梯度**：将多变量函数的所有偏导数打包成向量（如 \( \nabla f = (\frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \frac{\partial f}{\partial c}) \)），描述输入变量对输出的整体影响。

- **反向传播**：从输出节点开始，沿计算图反向遍历，通过「链式法则」传递梯度，自动计算所有输入/参数的梯度，是自动微分的核心。

- **链式法则**：若 \( f(g(x)) \)，则 \( f'(x) = f'(g(x)) \times g'(x) \)，反向传播本质是链式法则的逐层应用。

## 1.4 tanh 激活函数（与 tan 的区别）

- **tanh（双曲正切）**：神经网络专用激活函数，引入非线性，公式：\( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
        

    - 值域：[-1, 1]，平滑可导，无断点，适合梯度传递。

    - 导数公式：\( \tanh'(x) = 1 - \tanh^2(x) \)（简洁易计算，适配反向传播）。

- **tan（正切）**：三角函数，公式 \( \tan(x) = \frac{\sin x}{\cos x} \)，值域(-∞,+∞)，周期震荡、有断点，无法用于神经网络激活。

- **关键提醒**：tanh 与 tan 无任何关系，仅名字相似；tan 的反函数是 arctan，与 tanh 无关。

# 二、核心代码实现（完整可运行）

## 2.1 Value 类（自动微分核心）

封装「数值+梯度+计算图关系」，实现加法、乘法、tanh 运算及反向传播，是整个 micrograd 的核心。

```python
import math
import random

class Value:
    def __init__(self, data, _children=(), _op=''):
        # 核心属性
        self.data = data          # 节点的数值（前向传播结果）
        self.grad = 0.0           # 节点的梯度（反向传播结果，初始为0）
        self._prev = set(_children)# 该节点的子节点（计算图依赖关系）
        self._op = _op            # 该节点的运算符号（+、*、tanh等）
        self._backward = lambda: None  # 反向传播函数（初始化空函数）

    # 打印格式化，方便查看数值和梯度
    def __repr__(self):
        return f"Value(data={round(self.data, 6)}, grad={round(self.grad, 6)})"

    # 重载加法：实现 Value 对象的 + 运算
    def __add__(self, other):
        # 确保 other 是 Value 类型（兼容常数运算）
        other = other if isinstance(other, Value) else Value(other)
        # 前向传播：计算加法结果
        out = Value(self.data + other.data, (self, other), '+')
        
        # 反向传播：加法的梯度规则（两边梯度均等于输出梯度）
        def _backward():
            self.grad += out.grad  # 用 += 防止多用途节点梯度被覆盖
            other.grad += out.grad
        out._backward = _backward
        
        return out

    # 重载乘法：实现 Value 对象的 * 运算
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # 前向传播：计算乘法结果
        out = Value(self.data * other.data, (self, other), '*')
        
        # 反向传播：乘法的梯度规则（梯度=输出梯度×另一个乘数）
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        
        return out

    # 实现 tanh 激活函数
    def tanh(self):
        x = self.data
        # 前向传播：计算 tanh 值
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), 'tanh')
        
        # 反向传播：tanh 的导数规则
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    # 核心：自动反向传播（拓扑排序+链式法则）
    def backward(self):
        # 1. 拓扑排序：确保按「子节点→父节点」的顺序处理（避免梯度传递混乱）
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                # 先递归处理所有子节点
                for child in v._prev:
                    build_topo(child)
                # 子节点处理完，再加入当前节点
                topo.append(v)
        
        # 从当前节点（输出节点）开始构建拓扑序
        build_topo(self)
        
        # 2. 反向传播起点：输出节点对自身的梯度为1（dout/dout = 1）
        self.grad = 1.0
        
        # 3. 从后往前（反向拓扑序），逐个执行反向传播函数
        for node in reversed(topo):
            node._backward()

    # 重载减法、除法（可选，补充完整运算支持）
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1

```

## 2.2 单个神经元实现

神经元是神经网络的基本单元，核心逻辑：输入×权重 + 偏置 → tanh 激活输出。

```python
# 单个神经元实现（2输入，1输出）
def single_neuron_demo():
    # 1. 输入信号（x1, x2）
    x1 = Value(2.0)
    x2 = Value(0.0)
    
    # 2. 权重（w1, w2）：随机初始化（-1到1之间），代表输入的重要程度
    w1 = Value(-3.0)
    w2 = Value(1.0)
    
    # 3. 偏置（b）：调整神经元基准活跃度，避免模型表达能力不足
    b = Value(0.5)
    
    # 4. 前向传播：加权和 + tanh 激活
    wx_sum = x1 * w1 + x2 * w2 + b  # 加权和：w1x1 + w2x2 + b
    out = wx_sum.tanh()             # 激活输出：tanh(加权和)
    
    # 5. 反向传播：自动计算所有参数的梯度
    out.backward()
    
    # 打印结果
    print("=== 单个神经元输出 ===")
    print(f"输入 x1: {x1}, x2: {x2}")
    print(f"权重 w1: {w1}, w2: {w2}")
    print(f"偏置 b: {b}")
    print(f"神经元输出: {out}")

# 运行单个神经元示例
single_neuron_demo()

```

## 2.3 多层感知机（MLP）实现

MLP = 多层神经元堆叠（输入层→隐藏层→输出层），层与层全连接，实现复杂非线性拟合。

```python
class Neuron:
    """单个神经元类（复用单个神经元逻辑）"""
    def __init__(self, n_in):
        # n_in：输入维度（如2输入神经元，n_in=2）
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]  # 权重
        self.b = Value(random.uniform(-1, 1))                          # 偏置

    def __call__(self, x):
        # 前向传播：输入x → 加权和 → tanh激活
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

class Layer:
    """神经元层类（多个神经元组成一层）"""
    def __init__(self, n_in, n_out):
        # n_in：输入维度；n_out：当前层神经元数量
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        # 前向传播：输入x → 输出当前层所有神经元的结果
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs  # 单个输出直接返回值

class MLP:
    """多层感知机类（多层Layer堆叠）"""
    def __init__(self, dimensions):
        # dimensions：各层维度（如[2, 3, 1] → 输入2维，隐藏层3神经元，输出1维）
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(Layer(dimensions[i], dimensions[i+1]))

    def __call__(self, x):
        # 前向传播：输入x → 依次经过所有层 → 最终输出
        for layer in self.layers:
            x = layer(x)
        return x

# MLP 运行示例
def mlp_demo():
    # 1. 初始化MLP：输入2维，隐藏层3神经元，输出1维
    mlp = MLP([2, 3, 1])
    
    # 2. 输入数据（x1=1.0, x2=-2.0）
    x = [Value(1.0), Value(-2.0)]
    
    # 3. 前向传播：得到预测值
    y_pred = mlp(x)
    
    # 4. 反向传播：计算所有参数（权重、偏置）的梯度
    y_pred.backward()
    
    # 打印结果
    print("\n=== MLP 输出 ===")
    print(f"输入 x: {x}")
    print(f"MLP 预测输出: {y_pred}")
    # 查看隐藏层第一个神经元的第一个权重（示例）
    print(f"隐藏层第1个神经元的第1个权重: {mlp.layers[0].neurons[0].w[0]}")

# 运行MLP示例
mlp_demo()

```

## 2.4 神经网络训练（梯度下降）

训练核心：通过梯度下降更新参数（权重、偏置），最小化损失（预测值与真实值的误差）。

```python
def train_mlp():
    # 1. 初始化MLP
    mlp = MLP([2, 3, 1])
    # 2. 输入数据和真实标签（示例：输入[1.0, -2.0]，期望输出0.5）
    x = [Value(1.0), Value(-2.0)]
    y_true = Value(0.5)  # 真实标签
    learning_rate = 0.01  # 学习率：控制参数更新步长（避免过大/过小）
    epochs = 100  # 训练轮次：反复更新参数
    
    print("\n=== MLP 训练过程 ===")
    for epoch in range(epochs):
        # 3. 前向传播：得到预测值
        y_pred = mlp(x)
        
        # 4. 计算损失（MSE损失：均方误差，衡量预测值与真实值的差距）
        loss = (y_pred - y_true) * (y_pred - y_true)
        
        # 5. 重置所有梯度（避免上一轮梯度累积）
        for layer in mlp.layers:
            for neuron in layer.neurons:
                for w in neuron.w:
                    w.grad = 0.0
                neuron.b.grad = 0.0
        
        # 6. 反向传播：计算梯度
        loss.backward()
        
        # 7. 梯度下降：更新所有参数（w = w - 学习率×梯度，b同理）
        for layer in mlp.layers:
            for neuron in layer.neurons:
                for w in neuron.w:
                    w.data -= learning_rate * w.grad
                neuron.b.data -= learning_rate * neuron.b.grad
        
        # 每10轮打印一次损失（查看训练效果）
        if (epoch + 1) % 10 == 0:
            print(f"轮次 {epoch+1:3d} | 损失: {round(loss.data, 6)} | 预测值: {round(y_pred.data, 6)}")

# 运行训练示例
train_mlp()

```

# 三、关键疑问解答（高频问题）

## 3.1 _backward 前面的下划线是什么意思？

Python 约定俗成的命名规范：**单下划线开头 = 私有成员**。

- `_backward`：是内部梯度计算函数，由 `backward()` 方法自动调用，用户无需手动调用。

- `backward()`：是公开接口，用户直接调用（如 `out.backward()`），触发整个反向传播。

- 同类私有成员：`_prev`（存储子节点）、`_op`（存储运算符号），均为类内部使用，用户无需关注。

## 3.2 为什么梯度计算用 += 而不是 = ？

避免「同一节点被多次使用时，梯度被覆盖」。

例如：一个变量 x 同时参与两个运算（如 x*a 和 x*b），它会收到两份梯度，需要将两份梯度累加，才能得到 x 对最终输出的总影响。若用 =，会覆盖其中一份梯度，导致计算错误。

## 3.3 拓扑排序的作用是什么？

确保反向传播的顺序正确：**必须先处理父节点的梯度，再传递给子节点**（即「从输出到输入」的顺序）。

若没有拓扑排序，可能出现「子节点梯度未计算，就先计算父节点梯度」的错误，导致梯度传递混乱。

# 四、课程核心总结

## 4.1 核心逻辑链条

Value 类（封装数值+梯度）→ 运算重载（+、*、tanh）→ 计算图构建 → 拓扑排序 → 反向传播（链式法则）→ 神经元/MLP → 梯度下降训练。

## 4.2 关键收获

- 深度学习框架的底层核心是「自动微分」，micrograd 是极简实现（200行代码）。

- 反向传播的本质是「链式法则+拓扑排序」，自动计算所有参数的梯度。

- 神经网络的训练就是「前向预测→计算损失→反向求梯度→更新参数」的循环。

- 激活函数（如 tanh）的核心作用是「引入非线性」，让神经网络能拟合复杂规律。

## 4.3 延伸思考

- 可补充其他激活函数（如 ReLU）、损失函数（如交叉熵）。

- 可扩展到批量数据训练、学习率调整等优化策略。

- PyTorch/TensorFlow 的核心逻辑与 micrograd 一致，只是增加了工程优化（如GPU加速、批量运算）。

# 五、补充说明

本笔记代码可直接复制运行（Python 3.7+），无需额外依赖（仅用标准库 math、random）。

上传 GitHub 时，建议命名为 `micrograd_notes.md`，确保公式、代码块正常渲染（GitHub 支持 LaTeX 公式和 Python 代码高亮）。
> （注：文档部分内容可能由 AI 生成）