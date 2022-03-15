- [x] [Adversarial reciprocal points learning for open set recognition](https://arxiv.org/abs/2103.00953)[:page_facing_up:](C:\Users\smart-dsp\Zotero\storage\44BRUSHS\Chen 等。 - 2021 - Adversarial Reciprocal Points Learning for Open Se.pdf)

  > TPAMI 2021
  >
  > ARPL
  >
  > 在不损失已知分类精度的前提下，最小化已知分布和未知分布的重叠
  >
  > 通过生成混淆样本，来增强已知未知样本的能力

- 每个互反点由对应已知类别的类外空间学习，利用多个已知类别之间的对抗来降低经验分类风险。提出了一种对抗边界约束，通过限制由倒易点构成的潜在开放空间来降低开放空间风险。为了进一步估计开放空间的未知分布，基于互反点与已知类之间的对抗机制，设计了一种实例化的对抗增强方法，生成多样化且容易混淆的训练样本。这样可以有效地增强模型对未知类的可分辨性。

- 特征空间的划分

  - $\mathcal{D}_\mathcal{L}=\{(x_1,y_1),...,(x_n,y_n)\}$是来自N个类的n个有标签的样本

  - $\mathcal{D}_\mathcal{T}$是从$\{1,...,N\}\cup\{N+1,...,N+U\}$,开集中采集来的测试集样本，其中 $U$是未知类的数量

  - $\mathcal{S}_k$是第$k$类的$deep \ embedding \ space$

  - $\mathcal{O}_k$是在embedding space中除第$k$类的$\mathcal{S}_k$外的$space$
    $$
    \mathcal{O_k}=\mathcal{O_k^{pos}\cup O_k^{neg}}
    $$

    - $\mathcal{O_k^{pos}}$为除第$k$类外，其他known classes的$space$
    - $\mathcal{O_k^{neg}}$为unknown classes的$space$，潜在的未知类的空间

- 优化目标（二元任务）
  $$
  {\arg {min}}_{\psi_k}\{\mathcal{R^k|R_\epsilon(\psi_k,S_k\cup O_k^{pos})+\alpha\cdot R_o(\psi_k,O_k^{neg})}\}
  $$

  - $\mathcal{R^k}=\mathcal{R_\epsilon}+\alpha\cdot R_o$，为设定的预期误差
  - $\mathcal{R_\epsilon}$是一个empirical classification risk（经验分类风险），如cross entropy 
  - $\mathcal{\psi_k}:\R^d\to\{0,1\}$是一个网络将embedding $x$ 映射为label k（是label k 或 不是label k）
  - $\mathcal{R_o}$是$open\ space\ risk$
    - <img src="Open-Set Recognition.assets/image-20220305170259251.png" alt="image-20220305170259251" style="zoom:80%;" />
      - 分子为：unknown classes的样本经过网络在第$k$类上输出概率的积分
      - 分母为：known classes的样本经过网络在第$k$类上输出概率的积分，
      - unknown classes在类别输出上的概率最小化（距离各reciprocal point都近），让known classes的样本在类别输出上概率最大化（距离reciprocal point 远），由前一个约束共同作用可得，known classes中占比最大的应是label 为 $k$的样本
  - 目标：找到误差$\R^k$最小的网络$\psi_k$

- 优化目标（多分类任务）

  > 多分类任务拆为多个二元分类任务

  - $$
    \arg{\min}_{f\in \mathcal{H}}{R_\epsilon(f,\mathcal{D_L})+\alpha\cdot \sum_{k=1}^N{R_o(f,\mathcal{D}_U)}}\\
    $$

    - 其中，$f=\odot(\psi_1,\psi_2,...,\psi_k)$ ,是整合了k个分类的网络
    - $\mathcal{D}_\mathcal{L}=\{(x_1,y_1),...,(x_n,y_n)\}$是来自N个类的n个有标签的样本集
    - $\mathcal{D}_U$为unknown classes的样本空间



- Reciprocal Point

  > 用网络学一个距离自身类embedding feature最远的embedding feature，命名为Reciprocal Point，P
  >
  > 使用距离作为softmax层的输入，不再是两个向量（这一层权重，上一层神经网络的输出）的内积
  >
  > Loss为最大化输出类别的概率，起到了降低经验分类风险，实则优化了网络，使$P$距离$\mathcal{S_k}$进一步扩大，
  >
  > 开集分类SOTA

  - $P^k$被视为除自己类别外的训练数据子集和开集数据，即数据集$\mathcal{D_L^{\neq k}\cup D_U}$的特征表示，用$m$维的特征向量进行表示

  - - [ ] :question:（怎么实现的？）除k外的$embedding\ space$ :$\mathcal{O_k}$ ，应该比$k$类的特征向量$\mathcal{S_k}$ 距离$P^k$更近

    - <img src="Open-Set Recognition.assets/image-20220305185632482.png" alt="image-20220305185632482" style="zoom:80%;" />

  - 距离函数：欧式距离（空间）和点乘（角度）结合

    - <img src="Open-Set Recognition.assets/image-20220305185911590.png" alt="image-20220305185911590" style="zoom:80%;" />
      - 这里是网络对$x$的$embedding \ feature$和$reciprocal \ point$的距离
      - 欧式距离最大化，让$\mathcal{S_k,P_k}$“距离”最远
      - 夹角最小化，让同一类的feature聚合
        - [ ] :question:有没有对feature进行归一化？
    - x的预测类别判据为：距离$P^k$最远的那个类
      - <img src="Open-Set Recognition.assets/image-20220305190507874.png" alt="image-20220305190507874" style="zoom:80%;" />
    - 分类Loss
      - <img src="Open-Set Recognition.assets/image-20220305190713427.png" alt="image-20220305190713427" style="zoom:80%;" />
      - 等价于最大化$\mathcal{S_k}和P^k$的距离,最小化$\mathcal{S_k}和\mathcal{P^k}$的距离
        - <img src="Open-Set Recognition.assets/image-20220305190955888.png" alt="image-20220305190955888" style="zoom:80%;" />

- adversarial margin constraint term （对抗性边沿约束项）

  - 目的：

    - 将开放空间限制在有界范围内，来降低每个已知类别的开放风险

      <img src="Open-Set Recognition.assets/image-20220308101953542-16467062070032.png" alt="image-20220308101953542" style="zoom:80%;" />

    - 由于开放空间的样本不可能考虑完全，所以通过将$embedding\ space \ \mathcal{S_k}与\mathcal{P_k}$的距离限制为小于R来间接实现上述条件

      <img src="Open-Set Recognition.assets/image-20220309104440376.png" alt="image-20220309104440376" style="zoom:80%;" />

      - R是可学习的裕度
      - 只使用了欧式距离
      - [ ] :question:(通过限制已知类与P之间的关系如何约束到未知空间中？)
        - [ ] $P^k$是在$embedding\ space$中距离$k$类最远的相反类的特征，$P$点附近都为与$P$相似的未知类，当把多数$P$聚集在一起的时候达到未知类聚合的目的

  - 通过约束原型和reciprocal point将类外嵌入空间限制在一个有界范围内

    <img src="Open-Set Recognition.assets/image-20220305142412627-16467062015101.png" alt="image-20220305142412627" style="zoom:80%;" />

  - 每个已知类都属于其他类的额外类空间。在各类相互作用时，已知类的原型会被自身的reciprocal point和其他类约束到一个有界的范围内

    <img src="Open-Set Recognition.assets/image-20220305143848061.png" alt="image-20220305143848061" style="zoom: 80%;" />

  - 所有已知类分布在bounded embedding space的外侧，未知的样本局限于内部有界的空间中

  - 有界约束防止神经网络对未知样本产生任意高的置信度。

  - Train and setting

    - Loss：

      <img src="Open-Set Recognition.assets/image-20220309153042290.png" alt="image-20220309153042290" style="zoom:80%;" />

      - 超参：$\lambda$
      - 可学习参数：$\theta,P,R$
      - ![image-20220309153756272](Open-Set Recognition.assets/image-20220309153756272.png)

- Instantiated Adversarial Enhancement (实例化对抗增强机制)

  > 将开放空间约束在一个有界范围内（以known classes的embedding feature为”边界“的空间内）

  - <img src="Open-Set Recognition.assets/image-20220305191757581.png" alt="image-20220305191757581" style="zoom:80%;" />

  - 在生成器和鉴别器之间添加了一种对抗机制
    - 生成的样本应该让鉴别器认为是已知样本
    - 鼓励生成器生成的样本靠近reciprocal point
    - 意味着生成的样本在骗过鉴别器的同时，尽可能的靠近开放空间中未知部分（中间）

  - 网络架构

    <img src="Open-Set Recognition.assets/image-20220309155311960.png" alt="image-20220309155311960" style="zoom:80%;" />

    - Genertor的图像让Discriminator鉴别为已知类

      - 鉴别器（区分生成的和原始的）

        <img src="Open-Set Recognition.assets/image-20220309161001355.png" alt="image-20220309161001355" style="zoom:80%;" />

      - 生成器

        1. 骗过鉴别器

        <img src="Open-Set Recognition.assets/image-20220309161155462.png" alt="image-20220309161155462" style="zoom:80%;" />

        2. 生成的图像的在$embedding\ space$中应该接近所有的P（unknown类），通过分类器来辨别，实质上将提取后的特征和各个P点的距离求和取最小。
        
           <img src="Open-Set Recognition.assets/image-20220309161836292.png" alt="image-20220309161836292" style="zoom:80%;" />
        
           其中
        
           <img src="Open-Set Recognition.assets/image-20220309161853023.png" alt="image-20220309161853023" style="zoom:80%;" />
        
        3. 生成器总目标
        
           <img src="Open-Set Recognition.assets/image-20220309162917965.png" alt="image-20220309162917965" style="zoom:80%;" />
        
           其中
           $$
           H(z_i,\mathcal{P})=-\frac{1}{N}\sum_{k=1}^NS(z_i,\mathcal{P^k})\cdot log(S(z_i,\mathcal{P^k}))
           $$
           

  - 分类器通过生成的confusing samples进行优化（不优化生成器，所以没有鉴别器那项），实质是让网络把生成的样本拉近P点

    <img src="Open-Set Recognition.assets/image-20220309170058215.png" alt="image-20220309170058215" style="zoom:80%;" />

    - 算法流程

      <img src="Open-Set Recognition.assets/image-20220309171208172.png" alt="image-20220309171208172" style="zoom:80%;" />

  - [ ] :question:未知类预测选用什么标准