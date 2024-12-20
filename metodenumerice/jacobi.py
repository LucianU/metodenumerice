import numpy as np

from .error import increment, residual

from .jor import jor


def jacobi(A, b, x0, tol=1e-6, max_iter=100, opt=0):
    return jor(A, b, x0, 1, tol, max_iter, opt)


def jacobi_base(A, b, x0, tol=1e-6, max_iter=100, opt=0):
    x = x0.copy()
    d = np.diag(A)
    D = np.diag(d)
    L_U = A - D

    D_inv = np.diag(1 / d)

    for k in range(max_iter):
        x_new = D_inv @ (b - L_U @ x)

        if opt == 0:
            error = increment(x_new, x)
        elif opt == 1:
            error = residual(A, x_new, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            print(f"Jacobi: Converged in {k + 1} iterations.")
            return x_new, k + 1

        x = x_new
    else:
        print("Jacobi: Didn't converge.")
    return x, max_iter

