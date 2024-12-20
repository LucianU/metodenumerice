import numpy as np


def increment(x_new, x_old, norm=np.inf):
    step = np.linalg.norm(x_new - x_old, ord=norm)
    return step


def residual(A, x, b, norm=2):
    residual = np.linalg.norm(b - A @ x, ord=norm)
    return residual


def print_error(step, error):
    print("Step {} Error {:10.6g}".format(step, error))
