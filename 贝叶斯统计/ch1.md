# 经典统计学
> 基本观点：把数据（样本）看成是来自具有一定概率分布的总体，所研究的对象是这个总体而不局限于数据本身
- 总体分布：总体分布或总体所属分布族给我们的信息
- 样本信息：从总体抽取的样本给我们的信息

# 贝叶斯统计学
> 基本观点：任一个未知量$\theta$都可看作一个随机变量，应该用一个概率分布（先验分布）去描述对$\theta$的未知状况。

基于总体分布、样本信息、先验信息
- 先验信息：在抽样之前有关统计问题的一些信息。
- 先验分布：通过先验信息获得的分布

# 贝叶斯公式

## 密度函数形式
1. $p(x|\theta)$:随机变量$\theta$给定某个值时，总体指标$X$的条件分布
2. 根据参数$\theta$的先验信息确定<strong style="color : red">先验分布$\pi(\theta)$</strong>
3. 联合密度函数(似然函数)：$$
L(\theta')=p(\boldsymbol{x}|\theta')=\prod_{i=1}^np(x_i|\theta')
$$
> 这个联合密度函数综合了总体信息和样本信息
    ***样本 $\boldsymbol{x}=(x_1,...,x_n)$ 的产生***
    1. 从先验分布$\pi(\theta)$产生一个样本$\theta'$
    2. 从总体分布$p(x|\theta')$产生一个样本 $\boldsymbol{x}=(x_1,...,x_n)$
4.  $\theta'$满足先验分布$\pi(\theta)$,考虑$\theta'$的概率，得样本$x$和参数$\theta$得联合分布$$
h(\boldsymbol{x},\theta)=p(\boldsymbol{x}|\theta)\pi(\theta)
$$
5. 根据$\boldsymbol{x}=(x_1,...,x_n),h(\boldsymbol{x}|\theta)$对参数$\theta$做出统计推断——后验分布:
首先分解$h(\boldsymbol{x},\theta)$:$$
h(\boldsymbol{x},\theta)=\pi(\theta|\boldsymbol{x})m(\boldsymbol{x})
$$
其中$m(x)$是$\boldsymbol{x}$的边缘密度函数。$$
m(\boldsymbol{x})=\int_\Theta h(\boldsymbol{x},\theta)d\theta = \int_{\Theta} p(\boldsymbol{x}|\theta)\pi(\theta)d\theta
$$
贝叶斯公式的密度函数形式。在样本$\boldsymbol{x}$给定下，$\theta$的条件分布被称为$\theta$的<strong style="color:red">后验分布</strong>
$$
\pi(\theta|\boldsymbol{x})=\frac{h(\boldsymbol{x},\theta)}{m(\boldsymbol{x})}=\frac{p(\boldsymbol{x}|\theta)\pi(\theta)}{\int_\Theta p(\boldsymbol{x}|\theta)\pi(\theta)d\theta}
$$
6. 在$\theta$是离散随机变量时，先验分布可用先验分布列$\pi(\theta_i)，i=1,2,...$表示，后验分布：$$
\pi(\theta_i|\boldsymbol{x})=\frac{p(\boldsymbol{x}|\theta_i)\pi(\theta)}{\underset{j}{\sum} p(\boldsymbol{x}|\theta_j)\pi(\theta_j)}
,i=1,2,...$$