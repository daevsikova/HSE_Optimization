import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler


class Oracle:
    def __init__(self, X, y):
        self._X = X
        self._y = y
        self.N = self._X.shape[0]
        self.num_features = self._X.shape[1]
        self._num_calls = 0

    def y_pred(self, z):
        return expit(z)

    def value(self, w):
        self._num_calls += 1
        prob = self.y_pred(self._X @ w)
        return -1 / self.N * (self._y.T @ np.log(prob + 1e-16) + (1 - self._y.T) @ np.log(1 - prob + 1e-16))

    def grad(self, w):
        self._num_calls += 1
        return -1 / self.N * self._X.T @ (self._y - self.y_pred(self._X @ w))

    def hessian(self, w):
        self._num_calls += 1
        prob = self.y_pred(self._X @ w)
        return 1 / self.N * self._X.T @ np.diagflat(np.multiply(prob, (1 - prob))) @ self._X

    def hessian_vec_product(self, w, d):
        self._num_calls += 1
        prob = self.y_pred(self._X @ w)
        res1 = self._X @ d
        res2 = np.diagflat(prob * (1 - prob)) @ res1
        res3 = self._X.T @ res2
        return 1 / self.N * res3

    def fuse_value_grad(self, w):
        self._num_calls -= 1
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        self._num_calls -= 2
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        self._num_calls -= 2
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)

    def _clear_num_calls(self):
        self._num_calls = 0

def make_oracle(data_path, format='libsvm'):
    if format == 'libsvm':
        x, y = load_svmlight_file(data_path)

    elif format == 'tsv':
        data = np.genfromtxt(fname=data_path)
        x, y = data[:, 1:], data[:, 0]

    x = hstack((x, np.ones((x.shape[0], 1)).reshape(-1, 1)))
    # scaler = StandardScaler(with_mean=False).fit(x)
    # x = scaler.transform(x)
    y = np.where(y == 2, 0, np.where(y == 4, 1, y))
    y = np.where(y == -1, 0, y)
    y = y.reshape(-1, 1)

    return Oracle(x, y)
