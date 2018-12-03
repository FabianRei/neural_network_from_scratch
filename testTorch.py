from torchNet import TorchNet
from readDataFromPickle import loadMnist
from numpyNet import NumpyNet
import numpy as np

Net = TorchNet(784, 10)
x_train, y_train, x_test, y_test = loadMnist('mnist.pkl')

Net.setLr(0.001)
Net.trainNet(x_train, y_train, epochs=1)
Net.setLr(0.0001)
Net.trainNet(x_train, y_train, epochs=1)
MyNet = NumpyNet(784, 10)
MyNet.load_weights(Net.fc1.weight, Net.fc1.bias, Net.fc2.weight, Net.fc2.bias)
print(y_train[-3:])
y_pred = MyNet.forward(x_train[-30:])
print(f'my Net predicts {np.argmax(y_pred, 1)}')
print('nice')


