# 深度学习
## 概念
- Sigmoid Function
$$
\begin{aligned}
y=c\frac{1}{1+e^{-(b+wx_1)}}\\
=c·sigmoid(b+wx_1)    
\end{aligned}
$$

```python {cmd=true matplotlib=true}
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-50,50,1,dtype='float')
c = 10
b = 0.1
w = 0.2
y = c * 1/(1+np.exp(-(b+w*x)))
plt.plot(y)
plt.show() # show figure
```
