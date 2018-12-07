from TorchNet import TorchNet
from readDataFromPickle import loadMnist
from NumpyNet import NumpyNet
import numpy as np

Net = TorchNet(784, 10)
x_train, y_train, x_test, y_test = loadMnist('mnist.pkl')
# normalization
x_train = x_train/255
x_test = x_test/255
Net.setLr(0.1)
Net.trainNet(x_train, y_train, epochs=1)
Net.setLr(0.01)
Net.trainNet(x_train, y_train, epochs=1)
MyNet = NumpyNet(784, 10)
MyNet.loadWeights(Net.fc1.weight, Net.fc1.bias, Net.fc2.weight, Net.fc2.bias)
print(y_test[-30:])
y_pred = MyNet.train(x_test[-30:], y_test[-30:])
print(f'my Net predicts {np.argmax(y_pred, 1)}')
print('nice')


