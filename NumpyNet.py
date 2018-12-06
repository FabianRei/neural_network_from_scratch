import numpy as np


def logSoftmax(x):
    return np.log(np.exp(x)/np.sum(np.exp(x), axis=0))


class NumpyNet:
    def __init__(self, dimIn, dimOut, middleLayer=100):
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.middleLayer = middleLayer
        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0
        self.initialize()
        self.activationList1 = []
        self.activationList1_relu = []
        self.activationList2 = []

    def initialize(self):
        """
        I initially thought to use xavier here (numbers being multiplied by sqrt(1/Nin), but
        ReLU layers usually don't need this, as they aren't subject to the vanishing gradient problem.
        The best practice seems to multiply the random normal initialization by sqrt(2/Nin), so I'll use that
        instead of a plain normal distribution.
        Biases are initialized with 0.01 to ensure firing of ReLU at the beginning of training (this is best practice).
        """
        self.w1 = np.random.randn(self.middleLayer, self.dimIn) * np.sqrt(2/self.dimIn)
        self.b1 = np.full(self.middleLayer, 0.01)
        self.w2 = np.random.randn(self.dimOut, self.middleLayer) * np.sqrt(2/self.middleLayer)
        self.b2 = np.full(self.dimOut, 0.01)

    def loadWeights(self, w1, b1, w2, b2):
        self.w1 = w1.detach().numpy()
        self.b1 = b1.detach().numpy()
        self.w2 = w2.detach().numpy()
        self.b2 = b2.detach().numpy()

    def forward(self, x, trainMode=False):
        x = x.transpose()
        activation1 = np.matmul(self.w1, x) + np.expand_dims(self.b1, 0).transpose()
        activation1_relu = np.maximum(activation1, 0)
        activation2 = np.matmul(self.w2, activation1_relu) + np.expand_dims(self.b2, 0).transpose()
        if trainMode:
            self.activationList1.append(activation1)
            self.activationList1_relu.append(activation1_relu)
            self.activationList2.append(activation2)
        return logSoftmax(activation2).transpose()

    def zeroGrad(self):
        self.activationList1 = []
        self.activationList1_relu = []
        self.activationList2 = []

    def train(self, x, y, batchSize=1, shuffle=True):
        xEpoch = x
        yEpoch = y
        if shuffle:
            shuffler = np.random.permutation(len(x))
            xEpoch = xEpoch[shuffler]
            yEpoch = yEpoch[shuffler]
        numData = len(xEpoch)
        for j in range(int(numData / batchSize) + 1):
            start = j * batchSize
            end = min(start + batchSize, numData)
            if start == end:
                continue
            currData = xEpoch[start:end]
            currLabels = yEpoch[start:end]
            self.zeroGrad()
            pred = self.forward(currData, trainMode=True)
            loss = self.getNllLoss(pred, currLabels)



    @staticmethod
    def getNllLoss(pred, target):
        return -np.mean(pred[np.arange(len(pred)), target])

