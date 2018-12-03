import numpy as np


def logSoftmax(x):
    return np.log(np.exp(x)/np.sum(np.exp(x), axis=0))


class NumpyNet:
    def __init__(self, dimIn, dimOut):
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0

    def load_weights(self, w1, b1, w2, b2):
        self.w1 = w1.detach().numpy()
        self.b1 = b1.detach().numpy()
        self.w2 = w2.detach().numpy()
        self.b2 = b2.detach().numpy()

    def forward(self, x):
        print('debug')
        x = x.transpose()
        activation1 = np.matmul(self.w1, x) + np.expand_dims(self.b1, 0).transpose()
        activation1_relu = np.maximum(activation1, 0)
        activation2 = np.matmul(self.w2, activation1_relu) + np.expand_dims(self.b2, 0).transpose()
        return logSoftmax(activation2).transpose()

    @staticmethod
    def getNllLoss(pred, target):
        return -np.mean(pred[np.arange(len(pred)), target])

