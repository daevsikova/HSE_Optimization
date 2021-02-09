from oracle import make_oracle
import numpy as np


def test_grad(oracle):
    n = oracle.num_features
    eps = 1e-5
    results = []

    # 10 different points w
    for _ in range(10):
        w = np.random.uniform(-1, 1, size=n).reshape((-1, 1))

        # calculate grad
        grad_test = oracle.grad(w)

        # calculate finite-diff grad
        grad_true = np.zeros((n, 1))
        for i in range(n):
            e = np.zeros((n, 1))
            e[i] = eps
            f_right = oracle.value(w + e)
            f_left = oracle.value(w - e)
            grad_true[i] = (f_right - f_left) / (2 * eps)

        # check if grad_test is close to grad_true
        check_close = np.allclose(grad_true, grad_test, rtol=eps)
        results.append(check_close)
    print('Results of calc gradient for 10 random datapoints via func np.allclose():')
    return results


def test_hessian(oracle):
    n = oracle.num_features
    eps = 1e-5
    results = []

    # 10 different points w
    for _ in range(10):
        w = np.random.normal(0, 1, size=n).reshape((-1, 1))
        hessian_test = oracle.hessian(w)
        hessian_true = np.zeros((n, n))

        # generate basis - vectors e_i
        for i in range(n):
            e = np.zeros((n, 1))
            e[i] = 1

            # numerical approximation of hessian
            f_right = oracle.grad(w + eps * e)
            f_left = oracle.grad(w - eps * e)
            hessian_true[:, i] = ((f_right - f_left) / 2 / eps).reshape(-1)

        # check if hessian_test is close to hessian_true
        check_close = np.allclose(hessian_true, hessian_test, rtol=eps)
        results.append(check_close)
    print('Results of calc Hessian for 10 random datapoints via func np.allclose():')
    return results
