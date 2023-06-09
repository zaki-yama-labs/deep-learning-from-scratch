import pickle
import sys, os

from sigmoid import sigmoid
from softmax import softmax

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100  # バッチの数
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    # batch_size x 784
    x_batch = x[i:i+batch_size]

    # batch_size x 10
    y_batch = predict(network, x_batch)

    # batch_size x 1
    p = np.argmax(y_batch, axis=1)  # 最も確率の高い要素のインデックスを取得

    # p == t[i:i+batch_size] は
    # batch_size x 1 の [True True False True ...] みたいなベクトル
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
