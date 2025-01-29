import numpy as np

from .error import increment, residual


def jor(A, b, x0, omega, tol=1e-8, max_iter=1000, opt=0):
    D_inv = np.diag(1 / np.diag(A))  # D^(-1)
    L_U = A - np.diag(np.diag(A))   # L + U
    x = x0.copy()

    for k in range(max_iter):

        x_new = omega * D_inv @ (b - L_U @ x) + (1 - omega) * x

        if opt == 0:
            error = increment(x_new, x)
        elif opt == 1:
            error = residual(A, x_new, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            print(f"Converged in {k+1} iterations.")
            return x, k+1

        x = x_new.copy()
    else:
        print("Didn't converge.")
    return x

