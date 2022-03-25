# 代码相关构成

## torch

- ```python
  torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
  ```

  - 其中k是保留的k个值，largest=True意味着选取最大的，sorted=True是指将返回结果排序

  - topk返回的是一个tuple，第一个元素指返回的具体值，第二个元素指返回值的index

- 出现Nan的原因
  - 网络没有归一化
    - 使用x = F.normalize(x, dim=1)   
  - 运算中出现负数开方
  - 分母有0
- 

FAISS

- 
