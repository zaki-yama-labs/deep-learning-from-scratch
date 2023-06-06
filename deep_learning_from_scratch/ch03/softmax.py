import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# a = np.array([0.3, 2.9, 4.0])
# exp_a = np.exp(a)  # 指数関数
# print(exp_a)

# sum_exp_a = np.sum(exp_a)  # 指数関数の和
# print(sum_exp_a)


# y = exp_a / sum_exp_a
# print(y)
