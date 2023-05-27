import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    y = x > 0  # [False, True, True] などのbool値の配列を返す
    return y.astype(int)  # bool値をint型に変換
    # return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
plt.show()
