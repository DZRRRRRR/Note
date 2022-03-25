# Fista

[:page_facing_up:](C:\Users\smart-dsp\Documents\WeChat Files\liketomeet\FileStorage\File\2021-12\讲稿.pdf)

## 线性回归

给定数据集$D=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$,其中$x_i=(x_{i1};x_{i2};...;x_{xp}),y_i\in \R$为单个数据，及其向量展开（p个维度）。“线性回归”视图学一个线性模型尽可能准确地预测实值输出标记。

- $X$为N行p列的矩阵，其中有N个样本，每个样本有p个维度（特征），$x_i\in R^p,(i=1,...,N)$，矩阵表示如下
  $$
  X=(x_1,x_2,...,x_N)^T=\left [\begin{matrix}x_{11} & x_{12}& ... & x_{1p}\\x_{21} & x_{22} & ... & x_{2p}\\... & ... & ... & ...\\x_{N1} & x_{N2} & ... & x_{Np}\end{matrix}\right]_{N\times p}
  $$

- $Y$为N行1列的矩阵，共有N个样本。$y_i\in R,(i=1,2,...,N)$,$Y$的矩阵表示如下：
  $$
  Y=\left [\begin{matrix}y_1\\y_2\\...\\y_N \end{matrix}\right]_{N\times1}
  $$
  

$$
y=x\beta+b \ \ when \  y\in R^n,x\in R^{n \times p}
$$

- $y$为输出

- $x$为输入

- $\beta$为线性变换，需要拟合的参数

- $b$为噪声，观测值偏离预测值得大小

  



- 模型
  $$
  f(x)=x^T\beta=\beta^Tx
  $$

### 最小二乘法

- [ ] [矩阵求导](https://zhuanlan.zhihu.com/p/24709748)

- 思想：使得重构误差$\|X^T\beta-y\|^2$最小：
  $$
  L:\hat{\beta}=\mathop{\arg\min}\limits_{\beta}\|X\beta-y\|\\
  =\mathop{\arg\min}\limits_{\beta}(X\beta-y)^T(X\beta-y)\\
  =\mathop{\arg\min}\limits_{\beta}\ {E_\beta}
  $$

  - $X\in\R^{m\times n}，m行每行为一个n维的样本$
  - $y\in \R^{m\times 1},\beta\in \R^{n\times 1}$
  - $m$为样本数，$n$为特征数

- 求解

  对$E_\beta$求导：
  $$
  d{E_\beta}=(X d\beta)^T(X\beta-y)+(X\beta-y)^T(Xd\beta)\\
  =(X\beta-y)^T(Xd\beta)+(X\beta-y)^T(Xd\beta)\\
  =2(X\beta-y)^T Xd\beta
  $$
  由于

  - [ ] [矩阵微分与求导](https://zhuanlan.zhihu.com/p/90802394)

  $$
  dE_\beta=\frac{\partial{E_\beta}}{\partial \beta}^Td\beta
  $$

  得到 
  $$
  \frac{\partial E_\beta}{\part \beta}=2X^T(X\beta-y)
  $$
  

  - 如果$X$为非奇异矩阵,求极值
    $$
    \frac{\partial E_\beta}{\part \beta}=2X^T(X\beta-y)=0\\
    X^TX\beta=X^Ty\\
    \hat \beta = (X^TX)^{-1}X^Ty
    $$

  - 如果$X$为奇异矩阵，$X$不可逆，则需要其他方法

## 岭回归

- 多重共线性，

  - $(X^TX)$不为满秩，逆矩阵不存在，不能用最小二乘法的解析解

  - 多重共线性是一种统计现象，是指线性模型中的特征（解释变量）之间由于存在精确相关关系或高度相关关系，多重共线性的存在会使模型无法建立，或者估计失真。多重共线性使用指标方差膨胀因子(variance inflation factor,VIF)来进行衡量,通常当提到“共线性"，都特指多重共线性。

- 相关性
  - 衡量两个或多个变量一起波动的程度的指标，它可以是正的，负的或者0。当说变量之间具有相关性，通常是指线性相关性，线性相关一般由皮尔逊相关系数进行衡量，非线性相关可以使用斯皮尔曼相关系数或者互信息法进行衡量。

### Loss

$$
l(\beta)=\sum(x_i\beta-y)^2+\lambda\|\beta\|_2^2\\
=(X\beta-y)^T(X\beta-y)+\lambda\beta^T\beta
$$

对其求导：
$$
dl=(Xd\beta)^T(X\beta-y)+(X\beta-y)^T(Xd\beta)+2\lambda\beta^T d\beta\\
=2(X\beta-y)^TXd\beta+2\lambda\beta^T d\beta\\
=2[(X\beta-y)^TX+\lambda\beta^T]d\beta
$$

$$
\frac{\part l}{\part \beta}=2[(X\beta-y)^TX+\lambda\beta^T]^T
$$
求极值
$$
\frac{\part l}{\part \beta}=2[(X\beta-y)^TX+\lambda\beta]^T=0\\
X^T(X\beta-y)+\lambda\beta=0\\
X^TX\beta-X^Ty+\lambda\beta=0\\
\hat\beta=(X^TX+\lambda I)^{-1}X^Ty
$$

- $(X^TX+\lambda I)$保证了满秩，但$\beta$不再是无偏的了（:question:）
- 放弃无偏性，降低精度为代价