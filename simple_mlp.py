import numpy as np


def new_model(in_dim, h_dim, out_dim):
    w = {
        1: np.random.randn(in_dim, h_dim) / np.sqrt(in_dim),
        2: np.random.randn(h_dim, out_dim) / np.sqrt(h_dim)
    }

    b = {
        1: np.zeros(h_dim),
        2: np.zeros(out_dim)
    }

    return w, b


def relu(a):
    """
    Rectifier unit
    :param a: (array) activation vector.
    :return: (array) Relu activation.
    """
    a[a < 0] = 0
    return a


def sigmoid(a):
    """
    Sigmoid activation function.
    :param a: (array) activation vector.
    :return: (array) Sigmoid activation.
    """
    return 1 / (1 + np.exp(-a))


def diff_sigmoid(a):
    """
    Derivative of the sigmoid function.
    :param a: (array) activation vector.
    :return: (array)
    """
    return sigmoid(a) * (1 - sigmoid(a))


def diff_relu(a):
    """
    Derivative of the relu function.
    :param a: (array) activation vector.
    :return: (array)
    """
    a[a < 0] = 0
    a[a > 0] = 1
    return a


def feed_forward(p, w, b):
    """
    Feed forward propagation.
    :param p: (array) Parameters.
    :param w: (dict) Weights.
    :param b: (dict) Biases
    :return: (array) Output.
    """
    a = {}
    z = {}

    z[2] = np.dot(p, w[1]) + b[1]
    a[2] = relu(z[2])
    z[3] = sigmoid(np.dot(a[2], w[2]) + b[2])

    return a, z


def cost_mse(a, y):
    """
    Cost function.
    :param a: (array) Predictions
    :param y: (array) Ground truth labels
    :return: (flt) Loss
    """
    return 0.5 * np.sum((a - y)**2)


def diff_cost_mse(a, y):
    return a - y


def bpe_delta(a, y):
    """
    Back propagating error delta
    :param a: (array) Predictions
    :param y: (array) Ground truth labels
    :return: (array)
    """
    return diff_cost_mse(a, y) * diff_sigmoid(a)


class NeuralNetwork:
    def __init__(self, in_dim, h_dim, out_dim, learning_rate=1e-4):
        """
        Simple one hidden layer net with relu activation in the hidden layer and sigmoid activation at the output
        layer.

        :param in_dim: (int) Size of the input vector.
        :param h_dim: (int) No. of hidden nodes.
        :param out_dim: (int) No. of output nodes.
        :param learning_rate: (flt)
        """
        self.w, self.b = new_model(in_dim, h_dim, out_dim)
        self.x = None
        self.a = None  # activations
        self.z = None  # xi * wi + bi
        self.learning_rate = learning_rate

    def feed_forward(self, p):
        """
        Compute the activations and z's. z = w(x) + b
        :param p: (array)
        """
        self.x = p
        self.a, self.z = feed_forward(p, self.w, self.b)

    def backprop(self, labels):
        """
        Backpropagate the error and update the weights and biases

        :param labels: (array) Ground truth vector.
        """
        # partial derivative with respect to layer 2
        delta3 = bpe_delta(self.z[3], labels)

        # dc_db2 = delta3
        dc_dw2 = np.dot(self.a[2].T, delta3)

        # partial derivative with respect to layer 1
        delta2 = np.dot(delta3, self.w[2].T) * diff_relu(self.z[2])

        # dc_db1 = delta2
        dc_dw1 = np.dot(self.x.T, delta2)

        # update weights and biases
        self.w[2] -= self.learning_rate * np.mean(dc_dw2, 1)
        self.b[2] -= self.learning_rate * np.mean(np.mean(delta3, 1), 0)
        self.w[1] -= self.learning_rate * np.mean(dc_dw1, 1)
        self.b[1] -= self.learning_rate * np.mean(np.mean(delta2, 1), 0)

    def stats(self):
        """
        Prints some weights and biases
        """
        for i in range(1, 3):
            print("Weights layer {}:\n".format(i), self.w[i], "\nBiases layer {}:\n".format(i), self.b[i], "\n")

    def fit(self, x, labels, batch_size, epochs):
        """
        Train the net.

        :param x: (array) Input vector.
        :param labels: (array) Ground truth vector.
        :param batch_size: (int) Size of mini batch
        :param epochs: (int) No. of epochs to train.
        """

        for i in range(epochs):
            # Shuffle the data
            seed = np.arange(x.shape[0])
            np.random.shuffle(seed)
            x_ = x[seed]
            labels_ = labels[seed]

            for j in range(x.shape[0] // batch_size):
                self.feed_forward(x_[j * batch_size: (j + 1) * batch_size])
                self.backprop(labels_[j * batch_size: (j + 1) * batch_size])

            _, y = feed_forward(x, self.w, self.b)

            if i % epochs // 10 == 0:
                print("Loss:", cost_mse(y[3], labels))


if __name__ == "__main__":
    from sklearn import datasets
    import sklearn.metrics
    np.random.seed(1)

    # Load data
    data = datasets.load_iris()
    x = data["data"]
    x = (x - x.mean()) / x.std()
    y = np.expand_dims(data["target"], 1)

    # one hot encoding
    y = np.eye(3)[y]

    nn = NeuralNetwork(4, 8, 3, 2e-2)
    nn.fit(x, y, 10, int(1e3))

    # result
    _, y_ = feed_forward(x, nn.w, nn.b)
    y_true = []
    y_pred = []
    for i in range(len(y)):
        y_pred.append(np.argmax(y_[3][i]))
        y_true.append(np.argmax(y[i]))

    print(sklearn.metrics.classification_report(y_true, y_pred))

