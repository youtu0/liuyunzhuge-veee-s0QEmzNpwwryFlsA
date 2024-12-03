
#### 简介


在前两篇文章中，我们详细探讨了如何利用采样数据来估计回归曲线。接下来，在本节中，我们将深入讨论如何处理分类问题。


#### 章节安排


1. 背景介绍
2. 数学方法
3. 程序实现


## 背景介绍


### 线性可分




---


**线性可分**是指在多维空间RD中，对于任意两个类别的数据，总是存在一个超平面，可以将这两个类别的数据点完全分开。


在二分类问题中，如果数据集是线性可分的，那么可以找到一个超平面，使得屏幕的一侧的所有点属于一个类别，而另一侧的所有点都属于另一个类别。


设数据集D\={X,y}，其中X为输入特征向量，y为类别标签。


如果存在一个超平面L:z\=Xw\+b，使得：


∀yi\=1,xiw\+b\>0∀yi\=0,xiw\+b\<0则称该数据集X是线性可分的；其中w为权重向量，b为偏置项，xi为第i组数据，是矩阵X的第i行.


在一些情形下，并不是严格线性可分的，也就是说不存在一个超平面能够将所有不同类别的点完全分隔开来。这种情况下，我们可能会考虑使用“**宽松的线性可分**”（**Soft Margin**）的概念。
在宽松线性可分中，定义松弛变量ξ，原条件改为


∀yi\=1,xiw\+b\>−ξ∀yi\=0,xiw\+b\<ξ
> 注意到，对于任何一个超平面L，总是存在一个足够大的松弛变量ξ使得该超平面满足宽松线性可分条件。
> 因此，一般认为，对于某一个超平面Li，使得其满足宽松线性条件的最小常数ξi越小，则说明该直线划分效果越好。


### 初识激活函数




---


超平面L是一个从RD到R的映射（函数）。其值域为(−∞,∞)。然而，在实际应用中，通常希望输出的范围现在\[0,1]之间，以便于解释和处理。为了实现这一目标，通常会引入**激活函数**。


**Sigmoid**函数是一个经典的激活函数，因其连续性和较低的计算复杂度而在机器学习中得到了广泛的应用。Sigmoid 函数的定义如下：


σ(z)\=11\+e−z**主要特点**


1. 连续性和可导性：
Sigmoid 函数及其导数都是连续的，这使得它非常适合用于基于梯度下降的优化算法。
2. 输出范围：
Sigmoid 函数的输出范围是 ((0, 1\))，这使其在二分类问题中特别有用。它可以将线性组合的输出转换为一个概率值，从而更容易解释模型的预测结果。
3. 计算复杂度：
Sigmoid 函数的计算相对简单，不涉及复杂的数学运算，这有助于提高模型的训练速度。同时，其导函数可以方便的从原函数计算，即：σ′(z)\=σ(z)⋅(1−σ(z))


## 逻辑回归




---


逻辑回归（Logistic Regression）是一种广泛应用于分类问题的统计模型和机器学习算法。尽管名称中包含“回归”，但它实际上主要用于解决分类问题，特别是二分类问题。


### 工作原理




---


1. **线性组合**：
首先，逻辑回归模型对输入特征进行线性组合，也称对输入进行评估：


z\=wTx\+b
2. **Sigmoid变换**：
然后，将评估的结果z通过Sigmoid函数进行变换：


y^\=σ(z)\=11\+e−(wTx\+b)
> Sigmoid函数的输出 y^ 可以解释为样本属于正类的概率。
3. **决策边界**：
通常，选择一个阈值（例如0\.5）来决定分类结果：


如果如果如果如果y\={1如果 y^≥0\.50如果 y^\<0\.5


### 损失函数




---


逻辑回归的损失函数通常采用**对数损失（Log Loss）**或称交叉熵损失（Cross\-Entropy Loss）：


L(w,b)\=−1N∑i\=1N\[yilog⁡(y^i)\+(1−yi)log⁡(1−y^i)]其中，N是样本数量，yi是真实标签，y^i是预测的概率值。


### 优化




---


本文将介绍如何采用**梯度下降法**优化逻辑回归模型。
在梯度下降法中，核心的部分是计算损失LOSS关于参数w和b的梯度，其反应了参数更新的方向和步长。


通常采用链式法则计算梯度，以参数w为例，有：


∇wLOSS\=∂LOSS∂w\=∂LOSS∂y^∂y^∂z∂z∂w梯度下降法中，采用梯度的反方向作为更新方向，其公式为：


w:\=w−λ⋅∇wLOSS其中，λ为学习率。


## 程序实现




---


在上一篇文章《[机器学习：线性回归（下）](https://github.com)》中已经讲述了超平面L的实现方法；因此，本文中将讨论诸如**激活函数**、**对数损失**等上一章为设计的部分的程序实现。


### 激活函数




---


下述函数用于计算输入矩阵或向量的每个元素的Sigmoid函数值。



```


|  | MatrixXd Sigmoid::cal(const MatrixXd& input) { |
| --- | --- |
|  | return input.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); }); |
|  | } |


```

这段代码是一个简短的函数实现，代码解释如下：


1. **`input.unaryExpr`**：
`unaryExpr` 是Eigen库中的一个函数，用于对矩阵或向量的每个元素应用一个给定的单变量函数。在这里，`input` 是一个Eigen矩阵或向量。
2. **Lambda函数**：
`[](double x) { return 1.0 / (1.0 + exp(-x)); }` 是一个Lambda函数，它定义了一个匿名函数，接受一个 `double` 类型的参数 `x`，并返回 `1.0 / (1.0 + exp(-x))`。这个函数实现了Sigmoid函数的计算。



```


|  | MatrixXd Sigmoid::grad(const MatrixXd& input) { |
| --- | --- |
|  | Matrix temp = input.unaryExpr([](double x) { return 1.0 / (1.0 + exp(-x)); }); |
|  | return temp.cwiseProduct((1 - temp.array()).matrix()); |
|  | } |


```

这段代码实现了激活函数的梯度的计算，类似与`Sigmoid::cal()`，先计算激活函数σ(z)的值，再采用逐个元素相乘`cwiseProduct`计算（即Hadamard乘积）


A∘B\=\[a11a12a21a22]∘\[b11b12b21b22]\=\[a11⋅b11a12⋅b12a21⋅b21a22⋅b22]### 对数损失


下述函数分别采用**`Eigen`**的矩阵计算方法，实现了对数损失及对数损失的梯度的计算



```


|  | double LogisticLoss::computeLoss(const MatrixXd& predicted, const MatrixXd& actual) { |
| --- | --- |
|  | MatrixXd log_predicted = predicted.unaryExpr([](double p) { return log(p); }); |
|  | MatrixXd log_1_minus_predicted = predicted.unaryExpr([](double p) { return log(1 - p); }); |
|  |  |
|  | MatrixXd term1 = actual.cwiseProduct(log_predicted); |
|  | // MatrixXd term2 = (1 - actual).cwiseProduct(log_1_minus_predicted); |
|  | MatrixXd term2 = (1 - actual.array()).matrix().cwiseProduct(log_1_minus_predicted); |
|  |  |
|  | double loss = -(term1 + term2).mean(); |
|  |  |
|  | return loss; |
|  | } |
|  |  |
|  | MatrixXd LogisticLoss::computeGradient(const MatrixXd& predicted, const MatrixXd& actual) { |
|  | MatrixXd temp1 = predicted - actual; |
|  | MatrixXd temp2 = predicted.cwiseProduct((1 - predicted.array()).matrix()); |
|  |  |
|  | return (temp1).cwiseQuotient(temp2); |
|  | } |


```

为了便于读者理解、学习，下面给出了`LogisticLoss::computeLoss()`函数的标量方法实现（采用矩阵索引）：



```


|  | double computeLoss(const MatrixXd& predicted, const MatrixXd& actual) { |
| --- | --- |
|  | int n = predicted.rows(); |
|  | double loss = 0.0; |
|  | for (int i = 0; i < n; ++i) { |
|  | double p = predicted(i, 0); |
|  | double y = actual(i, 0); |
|  | loss += -(y * log(p) + (1 - y) * log(1 - p)); |
|  | } |
|  | return loss / n; |
|  | } |


```

 本博客参考[slower加速器](https://chundaotian.com)。转载请注明出处！
