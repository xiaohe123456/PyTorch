# tensors类似于Numpy的ndarrays, 同时tensors可以使用GPU进行计算

import torch

# 构造一个5*3的矩阵  不初始化
x = torch.empty(5, 3)
print(x)

# 构造一个随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 构造一个矩阵全为0  而且数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 构造一个张量， 直接使用数据
x = torch.tensor([5.5, 3])
print(x)

# 创建一个tensor 基于已经存在的tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
