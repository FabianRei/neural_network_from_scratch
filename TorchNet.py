import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np


class TorchNet(nn.Module):
    def __init__(self, dimIn, dimOut):
        super(TorchNet, self).__init__()
        self.fc1 = nn.Linear(dimIn, 100)
        self.fc2 = nn.Linear(100, dimOut)
        self.epochs = 2
        self.batchSize = 64
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def trainNet(self, x, y, epochs=None, batchSize=None, shuffle=True):
        if epochs is None:
            epochs = self.epochs
        if batchSize is None:
            batchSize = self.batchSize

        for i in range(epochs):
            xEpoch = Variable(torch.from_numpy(x).type(torch.float32))
            yEpoch = Variable(torch.from_numpy(y).type(torch.long))
            if shuffle:
                shuffler = np.random.permutation(len(xEpoch))
                xEpoch = xEpoch[shuffler]
                yEpoch = yEpoch[shuffler]
            numData = len(xEpoch)
            for j in range(int(numData / batchSize)):
                start = j * batchSize
                end = min(start + batchSize, numData)
                currData = xEpoch[start:end]
                currLabels = yEpoch[start:end]
                self.optimizer.zero_grad()
                netOut = self(currData)
                prediction = netOut.max(1)[1]
                loss = self.criterion(netOut, currLabels)
                loss.backward()
                self.optimizer.step()
                print(f'prediction/label:\n{prediction}\n{currLabels}\nloss is: {loss}\n')

    def setLr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def printLr(self):
        for g in self.optimizer.param_groups:
            print(g['lr'])
