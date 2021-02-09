'''
Запускать метод требуется из файла optimize.py
В данный файл вынесена реализация lbfgs без сбора статистики
(для лучшей читаемости кода, но не для запуска)
'''


class OptimizeLBFGS:
    def __init__(self):
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

            if norm_d / norm_d0 < tolerance:
                break
        return w
