import numpy as np


def dul_descomp(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    return D, U, L


def check_args(method):
    def wrapper(A, b, x0, tol, max_iter=100, opt=0):
        if opt not in [0, 1]:
            raise ValueError(f"Invalid `opt` value {opt}. Use 0 (increment) or 1 (residual)")
        return method(A, b, x0, tol, max_iter, opt)
    return wrapper

