import matplotlib.pyplot as plt
import numpy as np
# 随机一个数组在0到10之间
arr = np.random.randint(0, 10, 5)
print(arr)
# softmax函数


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# 输出softmax函数的结果
sa = softmax(arr)
print(sa)
# 输出softmax函数的结果的和
print(np.sum(sa))
# 画图
x = np.arange(1, 6, 1)
plt.plot(x, arr, 'b')
plt.plot(x, sa, 'r')
plt.show()
