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
        http://cs231n.github.io/linear-classify/#softmax
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


class MSE:
    def __int__(self, activation_fn=None):
        """

        :param activation_fn: Class object of the activation function.
        """
        if activation_fn:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = NoActivation

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_hat, y):
        """
        :param y_hat: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y - y_hat)**2)

    @staticmethod
    def prime(y_hat, y):
        return y - y_hat

    def delta(self, y_hat, y):
        self.prime(y_hat, y) * self.activation_fn.prime(y)


class NoActivation:
    @staticmethod
    def activation(z):
        """
        :param z: (array) w(x) + b
        :return: z (array)
        """
        return z

    @staticmethod
    def prime(x):
        """
        Linear relation. The prime is the input variable.
        z = w(x) + b
        z' = x
        :param x: (array) Input variable x
        :return: x: (array)
        """
        return x


class Network:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        Example of one hidden layer with
        - 2 inputs
        - 3 hidden nodes
        - 3 outputs


        layers -->    [1,        2,          3]
        ----------------------------------------

        dimensions =  (2,     3,          3)
        activations = (      Relu,      Sigmoid)
        """
        self.n_layers = len(dimensions)
        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a

if __name__ == "__main__":
    from sklearn import datasets
    import sklearn.metrics

    # Load data
    data = datasets.load_iris()
    x = data["data"]
    x = (x - x.mean()) / x.std()
    y = np.expand_dims(data["target"], 1)

    # one hot encoding
    y = np.eye(3)[y]

    nn = Network((4, 2, 2, 1), (Relu, Relu, Sigmoid))
    nn.feed_forward(x[:1])