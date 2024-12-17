import numpy as np

from .error import step, residual


def jor(A, b, x0, omega=1.0, tol=1e-6, max_iter=100, opt=0):
    """
    Jacobi Overrelaxation (JOR) with selectable error metric (OPT)
    """
    x = x0.copy()
    #D = np.diag(A)
    D_inv = np.diag(1 / np.diag(A))
    #R = A - np.diag(D)

    for k in range(max_iter):

        x_new = x + omega * (D_inv @ (b - A @ x) - x)
        #x_new = (1 - omega) * x + omega * (b - R @ x) / D

        if opt == 0:
            error = step(x_new, x)
        elif opt == 1:
            error = residual(A, x_new, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            return x_new, k + 1
        else:
            print("Step {} Error {:10.6g}".format(k + 1, error))
        x = x_new

    return x, max_iter


def jor_with_precompute(A, b, x0, omega=1.0, tol=1e-6, max_iter=100, opt=0):
    x = x0.copy()
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    D_inv = np.diag(1 / np.diag(D))
    c = D_inv @ b

    for k in range(max_iter):
        x_new = (1 - omega) * x + omega * (-D_inv @ (L + U) @ x + c)

        if opt == 0:  # Increment-based error
            error = np.linalg.norm(x_new - x, ord=np.inf)
        elif opt == 1:  # Residual-based error
            error = np.linalg.norm(b - (A @ x_new), ord=np.inf)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            return x_new, k + 1
        else:
            print("Step {} Error {:10.6g}".format(k + 1, error))
        x = x_new
