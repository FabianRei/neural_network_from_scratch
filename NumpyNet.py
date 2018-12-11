import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


class NumpyNet:
    def __init__(self, dimIn, dimOut, middleLayer=100, lr=0.1):
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.middleLayer = middleLayer
        self.lr = lr
        self.w1 = 0
        self.b1 = 0
        self.w2 = 0
        self.b2 = 0
        self.initialize()
        self.activation1 = 0
        self.activation1_relu = 0
        self.activation2 = 0
        self.activation2_softmax = 0
        self.curr_x = 0
        self.curr_y = 0
        self.loss = 0
        self.w1_gradients = 0
        self.b1_gradients = 0
        self.w2_gradients = 0
        self.b2_gradients = 0

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

    def load_torch_weights(self, w1, b1, w2, b2):
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

    def get_gradients(self):
        """ This is the hardest part.
        https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy is really helpful
        in understanding the topic. The way to go is to calculate the gradient for each forward pass activation.
        This then will be averaged and subtracted under perform_backprop.
         A great 'trick' or simplification in this case is that you can update the weights before the sigmoid by 
         multiplying the hidden layer activation with (softmax_output - target_onehot) of the output neuron it is linked to."""

        target_onehot = np.zeros((self.activation2_softmax.shape))
        target_onehot[np.arange(len(target_onehot)), self.curr_y] = 1
        softmax_minus_target = self.activation2_softmax - target_onehot
        ''' the next line takes advantage of numpy broadcasting. It makes sense to include the loss used for weight
        updates already in the gradient calculation step, which is also mathematically correct, if one assumes that
        these are the gradients of the averaged batch loss'''
        softmax_minus_target_mult_loss = softmax_minus_target * self.loss[:, np.newaxis]
        ''' multiply hidden layer activation with (softmax_output - target_onehot) for each combination, ergo each weight
        # sof[:, np.newaxis] @ act[np.newaxis, :] adds one axis to each vector to perform matrix multiplication
        # (10x1 * 1x784) '''
        self.w2_gradients = np.stack([sof[:, np.newaxis] @ act[np.newaxis, :] for act, sof in zip(self.activation1_relu.T, softmax_minus_target_mult_loss)], axis=0)
        self.b2_gradients = softmax_minus_target_mult_loss
        # Add up the gradients of all weights connected to the respective activation1 and ignore gradients, where
        # the activation is <= 0 to account for ReLU activation
        activation1_gradients = np.multiply(np.sum(np.transpose(self.w2_gradients, (0,2,1)), axis=2), (self.activation1>0).T)
        self.w1_gradients = np.stack(grad[:, np.newaxis] @ x_in[np.newaxis, :] for grad, x_in in zip(activation1_gradients, self.curr_x))
        self.b1_gradients = activation1_gradients
        
    def update_weights(self):
        w1_grad_avg = np.mean(self.w1_gradients, axis=0)
        b1_grad_avg = np.mean(self.b1_gradients, axis=0)
        w2_grad_avg = np.mean(self.w2_gradients, axis=0)
        b2_grad_avg = np.mean(self.b2_gradients, axis=0)
        self.w1 -= self.lr * w1_grad_avg 
        self.b1 -= self.lr * b1_grad_avg
        self.w2 -= self.lr * w2_grad_avg
        self.b2 -= self.lr * b2_grad_avg
        
    def train(self, x, y, batch_size=3, shuffle=True, epochs=1):
        x_epoch = x
        y_epoch = y
        if shuffle:
            shuffler = np.random.permutation(len(x))
            x_epoch = x_epoch[shuffler]
            y_epoch = y_epoch[shuffler]
        num_data = len(x_epoch)
        for i in range(epochs):
            for j in range(int(num_data / batch_size) + 1):
                start = j * batch_size
                end = min(start + batch_size, num_data)
                if start == end:
                    continue
                self.curr_x = x_epoch[start:end]
                self.curr_y = y_epoch[start:end]
                pred = self.forward(self.curr_x)
                self.loss = self.get_nll_loss(pred, self.curr_y)
                print(f'Prediction is {np.argmax(pred, axis=1)}\nGround truth is {self.curr_y}\nLoss is {np.mean(self.loss)}, accuracy is {np.mean(np.argmax(pred, axis=1) == self.curr_y)*100}%')
                self.get_gradients()
                self.update_weights()

    def set_lr(self, lr):
        self.lr = lr

    @staticmethod
    def get_nll_loss(pred, target):
        return -np.log(pred[np.arange(len(pred)), target])

