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
    def delta(y_true, y):
        """
        http://cs231n.github.io/linear-classify/#softmax
        https://stackoverflow.com/questions/27089932/cross-entropy-softmax-and-the-derivative-term-in-backpropagation
        :param y_true: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector.
        :return: (array) Delta vector.

        y are softmax probabilitys
        y_true is truth vector one hot encoded

        y         y_true
        [0.8]     [1]
        [0.1]     [0]
        [0.1]     [0]

        result:

        [-0.2]
        [0.1]
        [0.1]

        """
        return y - y_true

    @staticmethod
    def loss(y_true, y):
        """
        https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks

        :param y_true: (array) One hot encoded truth vector.
        :param y: (array) Prediction vector
        :return: (flt)
        """
        return -np.dot(y_true, np.log(y))


class MSE:
    def __init__(self, activation_fn=None):
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
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true)**2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


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
        self.loss = None
        self.learning_rate = None
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

    def back_prop(self, z, a, y_true):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # update weights and biases
        self.update_w_b(self.n_layers - 1, dw, delta)

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i])
            dw = np.dot(a[i - 1].T, delta)
            update_params[i - 1] = (dw, delta)

        for k, v in update_params.items():
            self.update_w_b(k, v[0], v[1])

    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        self.w[index] -= self.learning_rate * np.mean(dw, 1)
        self.b[index] -= self.learning_rate * np.mean(np.mean(delta, 1), 0)

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param loss: Loss class (MSE, CrossEntropy etc.)
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            y_ = y_true[seed]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self.feed_forward(x_[k:l])
                self.back_prop(z, a, y_[k:l])

            if (i + 1) % 100 == 0:
                print("Loss:", self.loss.loss(y_true, z[self.n_layers]))

if __name__ == "__main__":
    from sklearn import datasets
    #import sklearn.metrics
    np.random.seed(1)
    # Load data
    data = datasets.load_iris()
    x = data["data"]
    x = (x - x.mean()) / x.std()
    y = np.expand_dims(data["target"], 1)

    # one hot encoding
    y = np.eye(3)[y]

    nn = Network((4, 8, 3), (Relu, Sigmoid))

    nn.fit(x[:2], y[:2], MSE, 1, batch_size=2)
    #nn.fit(x, y, MSE, 10000, 16)