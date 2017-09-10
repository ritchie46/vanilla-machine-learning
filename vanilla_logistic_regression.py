import numpy as np

"""
# Logistic Regression

w^T · x = w0 + w1x1 + w2x2 + w3x3

P(A|x) = σ(w^Tx)

# Given data:

D = ((x1, y1), ..., (xn, yn))
x ∈ R
y ∈ {0, 1}


# Pro's:

- interpretable
- small # params (d + 1)
- computationally efficient
- extensible to multi-class

# Cons

- performance as good as other models

# Maximum Likelihood Estimation MLE

labels y
input  x

likelihood lw = P(y[i]| x[i], w) * P(y[j] | x[j], w) ... P(y[n] | x[n],w)

in product notation

lw = 
"""


def sigmoid(z):
    """
    :param z: w^Tx (scalar)
    :return: σ(w^Tx) (scalar)
    """
    return 1 / (1 + np.exp(-z))