import numpy as np


class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        z[z < 0] = 0
        z[z > 0] = 1
        return z


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))


class Softmax:
    @staticmethod
    def activation(z):
        """
        https://stackoverflow.com/questions/34968722/softmax-function-python

        Numerically stable version
        """
        e_x = np.exp(z - np.max(z))
        return e_x / e_x.sum()

    # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
    # http://cs231n.github.io/neural-networks-case-study/#loss


class CrossEntropy:
    """
    Used with Softmax activation in final layer
    """

    @staticmethod
    def activation(z):
        return Softmax.activation(z)

    @staticmethod
    def delta(y_hat, y):
        """
        https://stackoverflow.com/questions/27089932/cross-entropy-softmax-and-the-derivative-term-in-backpropagation
        :param y_hat: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector.
        :return: (array) Delta vector.

        y are softmax probabilitys
        y_hat is truth vector one hot encoded

        y         y_hat
        [0.8]     [1]
        [0.1]     [0]
        [0.1]     [0]

        result:

        [-0.2]
        [0.1]
        [0.1]

        """
        return y - y_hat

    @staticmethod
    def loss(y_hat, y):
        """
        https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks

        :param y_hat: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector
        :return: (flt)
        """
        return -np.dot(y_hat, np.log(y))


