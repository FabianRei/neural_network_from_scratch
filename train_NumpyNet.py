from read_data_from_pickle import load_mnist
from NumpyNet import NumpyNet
import numpy as np

x_train, y_train, x_test, y_test = load_mnist('mnist.pkl')
# normalization
x_train = x_train/255
x_test = x_test/255
Net = NumpyNet(dimIn=784, dimOut=10, middleLayer=512)
Net.set_lr(0.1)
Net.train(x_train, y_train, epochs=2, batch_size=200)
Net.set_lr(0.01)
Net.train(x_train, y_train, epochs=2, batch_size=200)
Net.set_lr(0.001)
Net.train(x_train, y_train, epochs=2, batch_size=200)
Net.set_lr(0.0001)
Net.train(x_train, y_train, epochs=2, batch_size=200)
print(y_test[-30:])
# y_pred = MyNet.train(x_test[-30:], y_test[-30:])
# print(f'my Net predicts {np.argmax(y_pred, 1)}')
print('nice')
