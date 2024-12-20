import numpy as np

from .error import increment, residual
from .utils import is_symm


def sgs(A, b, x0, tol=1e-8, max_iter=1000, op=0):
    if not is_symm(A):
        print("Matrix is not symmetric.")
        return (None, 0)

    n = len(b)
    x = x0.copy()
    D_inv = np.diag(1 / np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    for k in range(max_iter):
        x_old = x.copy()

        # Forward sweep
        for i in range(n):
            sigma = L[i, :] @ x + U[i, :] @ x
            x[i] = D_inv[i] * (b[i] - sigma)

        # Backward sweep
        for i in range(n - 1, -1, -1):
            sigma = L[i, :] @ x + U[i, :] @ x
            x[i] = D_inv[i] * (b[i] - sigma)

        # Check for convergence
        if op == 0:
            error = increment(x, x_old)
        elif op == 1:
            error = residual(A, x, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            print(f"SGS: Converged in {k+1} iterations.")
            return x, k+1

    else:
        print("SGS: Did not converge.")
    return x, max_iter

