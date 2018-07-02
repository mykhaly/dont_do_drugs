import numpy as np


class AF:
    @staticmethod
    def drelu(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    @staticmethod
    def softmax(w):
        e = np.exp(w - np.amax(w))
        dist = e / np.sum(e)
        return dist

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(y):
        return 1 - y * y
