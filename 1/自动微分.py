# audograd是PyTorch中所有神经网络的核心。autograd软件包为tensors上的所有操作提供自动微分。
# 它是一个由运行定义的框架，这意味着以代码运行的方式定义你的后向传播，并且每次迭代都可以不同。

# tensor的两个重要属性:
# .grad：记录梯度
# .grad_fn：记录操作

import torch

# 创建一个张量
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 针对张量 做加法
y = x + 2
print(y)

print(y.grad_fn)  # y为操作结果被创建 具有grad_fn

z = y * y * 3
print(z)
out = z.mean()
print(out)

# .requires_grad_( … ) 会改变张量的 requires_grad 标记。输入的标记默认为 False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)

a.requires_grad_(True)
print(a.requires_grad)

b = (a * a).sum()
print(b.grad_fn)


# 梯度  out.backward() 等同于 out.backward(torch.tensor(1.))
out.backward()
# 打印梯度  d(out)/dx

print(x.grad)

# 雅克比向量积 此时y不再是一个标量 torch.autograd 不能够直接计算整个雅可比
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# 计算雅克比向量积 只需要简单的传递向量给 backward 作为参数
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():

    print((x ** 2).requires_grad)






