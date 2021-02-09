import numpy as np
from scipy.optimize import line_search, brent
from scipy.linalg import cho_factor, cho_solve
import time


class LineSearchFactory:
    def create(self, method):
        if method == 'golden':
            return LineSearchGolden()
        elif method == 'wolfe':
            return LineSearchWolfe()
        elif method == 'armijo':
            return LineSearchArmijo()
        elif method == 'brent':
            return LineSearchBrent()
        elif method == 'lipschitz':
            return LineSearchLipschitz()
        else:
            raise ValueError(f'Unknown method: {method}')


class LineSearchGolden:
    def __init__(self):
        self.b = 20

    def find_alpha(self, f, w, direction):
        func = lambda alpha: f.value(w + alpha * direction)

        a = 1e-5
        b = self.b
        K = (np.sqrt(5) - 1) / 2
        I = K * (b - a)
        n = 0
        x_b, x_a = a + I, b - I
        f_a, f_b = func(x_a), func(x_b)

        while n < 1000 and I > 1e-8:
            I = K * I
            n += 1

            if f_a >= f_b:
                a = x_a
                x_a = x_b
                x_b = a + I
                f_a, f_b = f_b, func(x_b)

            else:
                b = x_b
                x_b = x_a
                x_a = b - I
                f_a, f_b = func(x_a), f_a

        if f_a <= f_b:
            return np.array(x_a)
        else:
            return np.array(x_b)


class LineSearchWolfe:
    def find_alpha(self, f, w, direction, c1=1e-4, c2=0.9, amax=None):
        alpha = line_search(f.value, lambda x: f.grad(x).reshape(-1), w, direction, c1=c1, c2=c2, amax=amax)[0]
        if not alpha:
            alpha = LineSearchArmijo().find_alpha(f, w, direction)
        return alpha


class LineSearchArmijo:
    def __init__(self):
        self.alpha_k = 1

    def find_alpha(self, f, w, direction, eta=0.5, c=0.0001):
        alpha = self.alpha_k
        f_w, grad = f.fuse_value_grad(w)
        grad_p = grad.T @ direction
        condition = lambda a: f.value(w + a * direction) <= f_w + c * a * grad_p
        while not condition(alpha):
            alpha = eta * alpha
        self.alpha_k = alpha / eta
        return alpha


class LineSearchLipschitz:
    def __init__(self):
        self.lipsch = 1

    def find_alpha(self, f, w, direction):
        l = self.lipsch
        f_w = f.value(w)
        dir_sq = direction.T @ direction
        betta0 = 2
        betta1 = 2

        while f.value(w + direction / l) > f_w - dir_sq / (2 * l):
            l *= betta0
        self.lipsch = l / betta1

        return 1 / l


class LineSearchBrent:
    def find_alpha(self, f, w, direction):
        func = lambda a: f.value(w + a * direction)
        alpha = brent(func)
        return alpha


class OptimizeGD:
    def __init__(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])

    def optimize(self, f, start_point, line_search_method='brent', tol=1e-8, max_iter=10000, **ls_params):
        f._clear_num_calls()
        self.clear_stats()

        t1 = time.time()  # for stats
        w = start_point
        func, d = f.fuse_value_grad(w)
        d = -d
        norm_d0 = np.linalg.norm(d) ** 2
        n = 0

        # logs
        self.num_iter = np.append(self.num_iter, n)
        self.time = np.append(self.time, 0.0)
        self.func_values = np.append(self.func_values, func)
        self.oracle_num_calls = np.append(self.oracle_num_calls, f._num_calls)
        self.grad_norm_k = np.append(self.grad_norm_k, norm_d0)

        # initialize line search method
        factory = LineSearchFactory()
        solver = factory.create(line_search_method)

        while n < max_iter:
            alpha = solver.find_alpha(f, w, d, **ls_params)
            w = w + alpha * d
            func, d = f.fuse_value_grad(w)
            d = -d
            norm_d = np.linalg.norm(d) ** 2
            n += 1

            t2 = time.time()
            # logs
            self.num_iter = np.append(self.num_iter, n)
            self.time = np.append(self.time, t2 - t1)
            self.func_values = np.append(self.func_values, func)
            self.oracle_num_calls = np.append(self.oracle_num_calls, f._num_calls)
            self.grad_norm_k = np.append(self.grad_norm_k, norm_d)

            if norm_d / norm_d0 < tol:
                break
        return w

    def clear_stats(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])


class OptimizeNewton:
    def __init__(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])

    def optimize(self, f, start_point, line_search_method='brent', tol=1e-5, max_iter=500):
        f._clear_num_calls()
        self.clear_stats()

        t1 = time.time()  # for stats
        w = start_point
        norm_d0 = np.linalg.norm(f.grad(w)) ** 2
        n = 0

        # initialize line search method
        factory = LineSearchFactory()
        solver = factory.create(line_search_method)

        # set tau for Hessian correction
        tau = 1e-4
        ident_mat = np.identity(w.shape[0])

        while n < max_iter:
            func, grad, hess = f.fuse_value_grad_hessian(w)
            grad = -grad
            norm_d = np.linalg.norm(grad) ** 2

            # logs
            t2 = time.time()
            self.num_iter = np.append(self.num_iter, n)
            self.time = np.append(self.time, t2 - t1)
            self.func_values = np.append(self.func_values, func)
            self.oracle_num_calls = np.append(self.oracle_num_calls, f._num_calls)
            self.grad_norm_k = np.append(self.grad_norm_k, norm_d)

            if norm_d / norm_d0 < tol:
                break

            while True:
                try:
                    c, low = cho_factor(hess)
                    break
                except np.linalg.LinAlgError:
                    hess = hess + tau * ident_mat
                    tau *= 2
            tau /= 10

            d = cho_solve((c, low), grad)
            norm = d.T @ d
            if norm ** 0.5 >= 1000:
                d = d / (norm ** 0.5)
            alpha = solver.find_alpha(f, w, d)
            
            w = w + alpha * d
            n += 1

        return w

    def clear_stats(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])


class OptimizeHFN:
    def __init__(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])

    def optimize(self, f, start_point, tol=1e-8, policy='sqrtGradNorm', tol_eta=0.5, max_iter=500):
        f._clear_num_calls()
        self.clear_stats()

        t1 = time.time()  # for stats
        w = start_point
        n = 0
        num_features = f.num_features
        solver = LineSearchFactory().create('wolfe')

        if policy == 'sqrtGradNorm':
            recalc_eps = lambda norm: min(0.5, norm ** 0.25) * norm ** 0.5
        elif policy == 'gradNorm':
            recalc_eps = lambda norm: min(0.5, norm ** 0.5) * norm ** 0.5
        elif policy == 'constant':
            recalc_eps = lambda norm: tol_eta * norm ** 0.5
        else:
            raise ValueError("Unknown optimization policy")

        while n < max_iter:
            func, r = f.fuse_value_grad(w)
            d = -r
            z = np.zeros((num_features, 1))
            rr_j = r.T @ r

            # logs
            t2 = time.time()
            self.num_iter = np.append(self.num_iter, n)
            self.time = np.append(self.time, t2 - t1)
            self.func_values = np.append(self.func_values, func)
            self.oracle_num_calls = np.append(self.oracle_num_calls, f._num_calls)
            self.grad_norm_k = np.append(self.grad_norm_k, rr_j)

            if rr_j <= tol:
                break

            eps = recalc_eps(rr_j)

            j = 0
            while j < 200:
                vec_pr = f.hessian_vec_product(w, d)
                dBd = d.T @ vec_pr

                if dBd <= 0:
                    if j == 0:
                        p = -r
                    else:
                        p = z
                    break

                a = rr_j / dBd
                z = z + a * d
                r = r + a * vec_pr
                rr_jk = r.T @ r
                if rr_jk ** 0.5 < eps:
                    p = z
                    break
                b = rr_jk / rr_j
                d = -r + b * d
                rr_j = rr_jk
                j += 1

            norm = p.T @ p
            if norm ** 0.5 >= 1000:
                p = p / (norm ** 0.5)

            alpha = solver.find_alpha(f, w, p)

            w = w + alpha * p
            n += 1

        return w

    def clear_stats(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])


class OptimizeLBFGS:
    def __init__(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])
        self.history = []
        self.sy_hist = []
        self.mu_hist = []


    def _recalc_d(self, d, n):
        hsize = min(n, self.s)
        sy_hist = [0] * hsize
        mu_hist = [0] * hsize

        ind = [hsize - k - 1 for k in range(hsize)]

        for i in ind:
            s, y = self.history[i]
            sy = s.T @ y
            mu = (s.T @ d) / sy
            sy_hist[i] = sy
            mu_hist[i] = mu
            d = d - mu * y

        if n > 0:
            y = self.history[hsize - 1][1]
            d = (sy_hist[hsize - 1] / (y.T @ y)) * d

        for i in ind:
            s, y = self.history[hsize - i - 1]
            sy = sy_hist[hsize - i - 1]
            mu = mu_hist[hsize - i - 1]
            betta = (y.T @ d) / sy
            d = d + (mu - betta) * s

        return d

    def lbfgs_optimize(self, f, init_point, tolerance=1e-8, history_size=10, max_iter = 1000):
        f._clear_num_calls()
        self.clear_stats()

        t1 = time.time()  # for stats

        w = init_point
        self.s = history_size
        func, grad = f.fuse_value_grad(w)
        norm_d0 = np.linalg.norm(grad) ** 2
        n = 0

        line_search_method = 'wolfe'
        factory = LineSearchFactory()
        solver = factory.create(line_search_method)

        while n < max_iter:
            d = self._recalc_d(-grad, n)
            alpha = solver.find_alpha(f, w, d)
            wk = w + alpha * d
            func, grad_k = f.fuse_value_grad(wk)

            if n > self.s:
                self.history.pop(0)

            self.history += [(wk - w, grad_k - grad)]
            norm_d = np.linalg.norm(grad_k) ** 2
            w, grad = wk, grad_k
            n += 1

            # logs
            t2 = time.time()
            self.num_iter = np.append(self.num_iter, n)
            self.time = np.append(self.time, t2 - t1)
            self.func_values = np.append(self.func_values, func)
            self.oracle_num_calls = np.append(self.oracle_num_calls, f._num_calls)
            self.grad_norm_k = np.append(self.grad_norm_k, norm_d)

            if norm_d / norm_d0 < tolerance:
                break
        return w

    def clear_stats(self):
        self.num_iter = np.array([])
        self.time = np.array([])
        self.func_values = np.array([])
        self.oracle_num_calls = np.array([])
        self.grad_norm_k = np.array([])
