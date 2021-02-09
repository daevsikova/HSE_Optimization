from numpy import sqrt, array


def golden_ratio_opt(f, a: float, b: float, eps: float = 1e-8):
    K = (sqrt(5) - 1) / 2
    I = K * (b - a)
    n = 0
    x_b, x_a = a + I, b - I
    f_a, f_b = f(x_a)[0], f(x_b)[0]

    while I >= eps:
        I = K * I
        n += 1

        if f_a >= f_b:
            a = x_a
            x_a = x_b
            x_b = a + I
            f_a, f_b = f_b, f(x_b)[0]

        else:
            b = x_b
            x_b = x_a
            x_a = b - I
            f_a, f_b = f(x_a)[0], f_a

    if f_a <= f_b:
        return array(x_a)
    else:
        return array(x_b)


def optimize(f, a: float, b: float, eps: float = 1e-8):
    x1, x3 = a, b
    x2 = (x1 + x3) / 2
    f1, f2, f3 = f(x1)[0], f(x2)[0], f(x3)[0]

    # if function isn't unimodal - call golden ratio search
    if f2 > f3 or f2 > f1:
        return golden_ratio_opt(f, a, b, eps)

    # for unimodal functions - parabolic method
    x = (x2 -
         ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1))
         /
         (2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)) + 1e-10))

    fx = f(x)[0]

    # new 3 dots
    if x1 <= x <= x2 and fx >= f2:  # x* in [x, x3]
        f1 = fx
        x1 = x

    elif x1 <= x <= x2 and fx < f2:  # x* in [x1, x2]
        f3 = f2
        x3 = x2
        x2 = x
        f2 = fx

    elif x2 <= x <= x3 and fx >= f2:  # x* in [x1, x]
        f3 = fx
        x3 = x

    elif x2 <= x <= x3 and fx < f2:  # x* in [x2, x3]
        f1 = f2
        x1 = x2
        f2 = fx
        x2 = x

    # new quadratic interpolation
    x_k = (x2 -
           ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1))
           /
           (2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)) + 1e-10))

    fk = f(x_k)[0]

    # check convergence
    while abs(x_k - x) > eps:
        x = x_k
        fx = fk

        if x1 <= x <= x2 and fx >= f2:  # x* in [x, x3]
            f1 = fx
            x1 = x

        elif x1 <= x <= x2 and fx < f2:  # x* in [x1, x2]
            f3 = f2
            x3 = x2
            x2 = x
            f2 = fx

        elif x2 <= x <= x3 and fx >= f2:  # x* in [x1, x]
            f3 = fx
            x3 = x

        elif x2 <= x <= x3 and fx < f2:  # x* in [x2, x3]
            f1 = f2
            x1 = x2
            f2 = fx
            x2 = x

        # new quadratic interpolation
        x_k = (x2 -
               ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1))
               /
               (2 * ((x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)) + 1e-10))

        fk = f(x_k)[0]

    return array(x_k)
