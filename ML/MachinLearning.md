# 机器学习
## 概念介绍
- Regression（回归）:输出是scalar
- Classification:分类
- Structured Learning:创造某种结构的事物（图片、文章等）
- Model：一个基于经验的带有未知参数的函数\
  1. Linear Model
  $$y=b+{\overset{N}{\underset{j=1}{\sum}}w_jx_j}$$
- Feature：$x_1$
- Weight：$w$
- Bias：$b$
- Loss: a funcion of parameters(w,b)\
how good a set of values is.\
常用的$Loss$形式
$$
Loss:\quad L=\frac{1}{N}\sum e_n  \\
MAE:\quad e=|y-\overline{y}|\quad\\
MSE:\quad e=(y-\overline{y})^2
$$
- Label:标签
### Hyperparameters(自己设定的参数)
- $\eta$:学习速率
- $w,b$:可随机选定
## 训练过程
### 写一个未知参数的函数（Model）
$$y=b+{\overset{N}{\underset{j=1}{\sum}}w_jx_j}$$
### Define Loss from Training Data
$$
Loss:\quad L=\frac{1}{N}\sum e_n  \\
MAE:\quad e=|y-\overline{y}|\quad
$$
### Optimization &nbsp; $w',b' = {\underset{w,b}{\operatorname{arg\,min}}\,L}$
#### Gradient Descent
1. pick an initial value $w^0,b^0$
2. Compute
$$
\eta \frac{\partial L}{\partial w}|_{w=w^0,b=b^0}\qquad  w^1\leftarrow w^0 - \eta\frac{\partial L}{\partial w}|_{w=w^0}
$$
$$
\eta \frac{\partial L}{\partial b}|_{w=w^0,b=b^0}\qquad  w^1\leftarrow w^0 - \eta\frac{\partial L}{\partial w}|_{w=w^0}
$$
3. Update $w$ iteratively

## 问题
1. 陷入局部解，没找到全局最优解

