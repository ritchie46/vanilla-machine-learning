from sklearn.utils.extmath import randomized_svd
import numpy as np
from functools import partial
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin


def np_svd(M, k):
    u, s, v = np.linalg.svd(M, full_matrices=False)
    return u[:, :k], s[:k], v[:k, :]


def inverse_svd(u, s, v):
    return np.dot(u * s, v)


def em_svd(X, n_iter, svd_f):
    nan_mask = np.isnan(X)

    # Fill NaNs in the columns first with means
    mu = np.nanmean(X, axis=0)
    X[nan_mask] = (mu[None, :] * np.ones(X.shape))[nan_mask]

    # Iteratively replace the NaN indexes with an SVD approximation
    # in an Expectation Maximization manner.
    for _ in range(n_iter):
        M_approx = inverse_svd(*svd_f(X))

        if np.allclose(X[nan_mask], M_approx[nan_mask], rtol=0.01, atol=0.01):
            break
        X[nan_mask] = M_approx[nan_mask]

    return X


class SVDImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_rows,
        k=4,
        n_iter_fit=20,
        n_iter_impute=5,
        svd_method="randomized",
        block_size=200,
    ):
        self.k = k
        self.svd_method = svd_method
        self.svd = (
            partial(np_svd, k=k)
            if svd_method == "numpy"
            else partial(randomized_svd, n_components=k)
        )
        self.n_rows = int(n_rows)
        self.n_iter_fit = int(n_iter_fit)
        self.n_iter_impute = n_iter_impute
        self.block_size = int(block_size)
        self.nan_filled_X = None

    def fit(self, X, y=None):
        assert self.n_rows < X.shape[0]
        X = np.array(X, dtype=np.float32)

        # Sort from least missing to most missing.
        idx = np.isnan(X).sum(axis=1).argsort()
        # Use only the least missing rows for fitting.
        X = X[idx[: self.n_rows], :]

        self.nan_filled_X = em_svd(X, self.n_iter_fit, self.svd)

    def transform(self, X, y=None):
        X = np.array(X, dtype=np.float32)

        for i in tqdm(range(X.shape[0] // self.block_size + 1)):
            start = i * self.block_size
            stop = start + self.block_size

            block = X[start:stop, :]
            M = np.concatenate([block, self.nan_filled_X])
            imputed = em_svd(M, n_iter=self.n_iter_impute, svd_f=self.svd)
            X[start:stop, :] = imputed[: block.shape[0], :]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)
