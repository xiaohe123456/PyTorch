# 神经网络可以通过torch.nn包来构建

# 一个典型的神经网络训练过程包括以下几点：
# 1.定义一个包含可训练参数的神经网络
# 2.迭代整个输入
# 3.通过神经网络处理输入
# 4.计算损失(loss)
# 5.反向传播梯度到神经网络的参数
# 6.更新网络的参数，典型的用一个简单的更新方法：weight = weight - learning_rate *gradient

# 定义神经网络

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        # 卷积
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 线性变换
        self.fc1 = nn.Linear(400, 400)
        self.fc2 = nn.Linear(400, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # 最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果池化层大小为方行 可以只写一个数
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):

        size = x.size()[1:]
        num_features = 1

        for s in size:

            num_features *= s

        return num_features

net = Net()
print(net)

# 一个模型可训练的参数可以通过调用 net.parameters() 返回

params = list(net.parameters())
print(len(params))
print(params[0].size())


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 所有参数梯度缓存器置零
net.zero_grad()

# 用随机的梯度来反向传播
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)                                             # MSELoss
print(loss.grad_fn.next_functions[0][0])                        # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])   # ReLU


net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 更新网络参数
# python 实现随机梯度下降
learning_rate = 0.01
for f in net.parameters():

    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# 创建优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)

loss.backward()
optimizer.step()
print(loss)
