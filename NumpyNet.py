import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


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
        self.activation1 = 0
        self.activation1_relu = 0
        self.activation2 = 0
        self.activation2_softmax = 0

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

    def forward(self, x):
        x = x.transpose()
        self.activation1 = np.matmul(self.w1, x) + np.expand_dims(self.b1, 0).transpose()
        self.activation1_relu = np.maximum(self.activation1, 0)
        self.activation2 = np.matmul(self.w2, self.activation1_relu) + np.expand_dims(self.b2, 0).transpose()
        self.activation2_softmax = softmax(self.activation2).transpose()
        return self.activation2_softmax

    def zeroGrad(self):
        self.activation1 = 0
        self.activation1_relu = 0
        self.activation2 = 0
        self.activation2_softmax = 0

    def get_gradients(self, target):
        """ This is the hardest part.
        https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy is really helpful
        in understanding the topic. The way to go is to calculate the gradient for each forward pass activation.
        This then will be averaged and subtracted under perform_backprop.
         A great 'trick' or simplification in this case is that you can update the weights before the sigmoid by 
         multiplying the hidden layer activation with (softmax_output - target_onehot) of the output neuron it is linked to."""

        target_onehot = np.zeros((self.activation2_softmax.shape))
        target_onehot[np.arange(len(target_onehot)), target] = 1
        softmax_minus_target = self.activation2_softmax - target_onehot
        # multiply hidden layer activation with (softmax_output - target_onehot) for each combination, ergo each weight
        w2_gradients = np.stack([sof[:, np.newaxis] @ act[np.newaxis, :] for act, sof in zip(self.activation1_relu.T, softmax_minus_target)], axis=0)
        b2_gradients = softmax_minus_target
        # Add up the gradients of all weights connected to the respective activation1 and ignore gradients, where
        # the activation is <= 0 to account for ReLU activation
        activation1_gradients = np.multiply(np.sum(np.transpose(w2_gradients, (0,2,1)), axis=2), (self.activation1>0).T)
        
        
        
        
        
        self.gradient2_softmax = gradient_matrix
        local_gradient2 = 0
        """ as all gradients in gradient2_softmax are zero except for one, we only need to look at that one combination.
        This reduces gradient2 to a formula with one multiplication except for a sun of 10 multiplications."""
        self.gradient2 = 0

    def train(self, x, y, batchSize=3, shuffle=True):
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
            pred = self.forward(currData)
            loss = self.getNllLoss(pred, currLabels)
            self.get_gradients(currLabels)



    @staticmethod
    def getNllLoss(pred, target):
        return -np.log(pred[np.arange(len(pred)), target])

