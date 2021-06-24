import numpy as np
from scipy.stats import beta
from scipy.special import gamma


def uniform_kernel(x1, x2, delta):
    """
    Default kernel suggested in (Rolland, ICML 2019)
    """
    u = np.abs(x1 - x2) / delta
    return int(u < 1)


def triang_kernel(x1, x2, delta):
    u = (x1 - x2) / delta
    return max(0, 1 - np.abs(u))


def quadr_kernel(x1, x2, delta):
    u = (x1 - x2) / delta
    return max(0, 15 * (1 - u ** 2) / 16)


def mirrored_kernel(x1, x2, delta, kernel=None, lb=0.0, ub=1.0):
    """
        Inputs: a kernel function and an evaluation points, smoothing delta
        Kwargs: lb, ub: lower/upped bound on interval. Default is [0, 1]
        Outputs: The mirrored kernel estimate (Schuster 1985, Equation 2.5)
    """
    return kernel(x1 + x2, 2 * lb, delta) + kernel(x1, x2, delta) + kernel(x1 + x2, 2 * ub, delta)


# Sample = X_i query_point = x
def beta_kernel(sample, query_point, delta):
    """
        Inputs: sample point, query point, bandwith delta
        Outputs: Beta kernel distance (centered at query_point, evaluated at sample)
        Reference: Chen 1999 "Beta kernel estimators for density functions"
    """
    a, b = query_point / delta + 1, (1 - query_point) / (delta) + 1

    if not (0 <= sample <= 1):
        return 0

    k = (sample) ** (query_point / delta) * (1 - sample) ** ((1 - query_point) / delta)
    b = gamma(a) * gamma(b) / gamma(a + b)

    # return beta.pdf(sample, a, b)
    return k / b


def modified_beta_kernel(sample, query_point, delta):
    """
        Inputs: sample point, query point, bandwith delta
        Outputs: Modified Beta kernel distance (centered at query_point, evaluated at sample)
        Reference: Chen 1999 "Beta kernel estimators for density functions"
    """
    if 0 <= query_point <= 2 * delta:
        a, b = p(query_point, delta), (1 - query_point) / delta
    elif 1 - 2 * delta <= query_point <= 1:
        a, b = query_point / delta, p(1 - query_point, delta)
    else:
        a, b = query_point / delta, (1 - query_point) / delta

    return beta.pdf(sample, a, b)


def p(x, b):
    return 2 * b ** 2 + 2.5 - np.sqrt(4 * b ** 4 + 6 * b ** 2 + 2.25 - x ** 2 - x / b)
