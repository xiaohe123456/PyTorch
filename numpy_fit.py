import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# 激活函数及其导数
def tanh(x):

    return np.tanh(x)

def tanh_derivative(x):

    return 1.0 - np.tanh(x) ** 2


# 初始化网络参数
input_size, hidden_size, output_size = 1, 10, 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

# 训练参数
learning_rate = 0.01
epochs = 300000

# 训练数据
X = np.linspace(-2, 2, 1000).reshape(-1, 1)
y = np.sin(X * 4)

# 训练循环
for i in range(epochs):
    # 前向传播
    z1 = np.dot(X, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = z2

    # 计算损失
    loss = np.mean((y_pred - y) ** 2)

    if i % 1000 == 0:

        print(f'Epoch {i}, Loss: {loss}')

    # 反向传播
    d_loss_y_pred = 2.0 * (y_pred - y) / y.size
    d_loss_W2 = np.dot(a1.T, d_loss_y_pred)
    d_loss_b2 = np.sum(d_loss_y_pred, axis=0)
    d_loss_a1 = np.dot(d_loss_y_pred, W2.T)
    d_loss_z1 = d_loss_a1 * tanh_derivative(z1)
    d_loss_W1 = np.dot(X.T, d_loss_z1)
    d_loss_b1 = np.sum(d_loss_z1, axis=0)

    # 参数更新
    W1 -= learning_rate * d_loss_W1
    b1 -= learning_rate * d_loss_b1
    W2 -= learning_rate * d_loss_W2
    b2 -= learning_rate * d_loss_b2

# 绘制结果
plt.plot(X, y, label='True')
plt.plot(X, y_pred, label='Predicted')
plt.legend()
plt.show()