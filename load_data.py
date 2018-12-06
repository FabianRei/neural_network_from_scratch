import pickle
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torch
from TorchNet import TorchNet

mnist = pickle.load(open('mnist.pkl', 'rb'))
x_train = mnist["training_images"]
y_train = mnist["training_labels"]
x_test = mnist["test_images"]
y_test = mnist["test_labels"]


# np.random.seed(42)
# selector = np.random.permutation(labels.shape[0])
# data = data[selector]
# labels = labels[selector]
# Image.fromarray(x_train[2].reshape(28,28)).show()




learningRate = 0.0001
Net = TorchNet(784, 10)
optimizer = optim.SGD(Net.parameters(), lr=learningRate)
criterion = nn.NLLLoss()


def train(epochs, batchSize, x, y, Net):
    for i in range(epochs):
        x = Variable(torch.from_numpy(x).type(torch.float32))
        y = Variable(torch.from_numpy(y).type(torch.long))
        numData = len(x)
        for j in range(int(numData/batchSize)):
            start = j*batchSize
            end = min(start+batchSize, numData)
            currData = x[start:end]
            currLabels = y[start:end]
            optimizer.zero_grad()
            netOut = Net(currData)
            prediction = netOut.max(1)[1]
            loss = criterion(netOut, currLabels)
            loss.backward()
            optimizer.step()
            print(f'prediction/label:\n{prediction}\n{currLabels}\nloss is :{loss}\n')
    return Net

Net.trainNet(x_train, y_train)
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
        return logSoftmax(activation2)


MyNet = NumpyNet(784, 10)

MyNet.load_weights(Net.fc1.weight, Net.fc1.bias, Net.fc2.weight, Net.fc2.bias)
print(y_train[-3:])
y_pred = MyNet.forward(x_train[-3:])
print(y_pred)
print('nice')
