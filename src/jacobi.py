import numpy as np
import numpy.linalg as nla

def jacobi_iterative(A, b, x0, TOL, max_it):  # a_ii!= 0 for all i
    n = A.shape[0]
    x = x0.copy()

    err = 1.0
    k_f = 0

    while k_f < max_it:
        for i in range(n):
            x_k = x.copy()
            temp_i_row = A[i, :].copy()
            temp_i_row[i] = 0.0
            x_k[i] = (-1) * (1 / A[i][i]) * (np.dot(temp_i_row, x_k) + b[i])
        err = nla.norm(x_k - x, np.inf) / nla.norm(x_k, np.inf)
        if err < TOL:
            x = x_k
            k_f += 1
            break
        else:
            x = x_k
            k_f += 1
    return x, err, k_f


def jacobi(A, b, x0, tol=1e-6, max_iter=100, opt=0):
    """
    Jacobi Method with selectable error metric (OPT)
    """
    x = x0.copy()
    d = np.diag(A)  # vector of diagonal
    D = np.diag(d)  # diagonal matrix
    R = A - D

    D_inv = np.diag(1 / d)

    for k in range(max_iter):
        x_new = (b - R @ x) @ D_inv

        if opt == 0:  # Increment-based error
            error = np.linalg.norm(x_new - x, ord=np.inf)
        elif opt == 1:  # Residual-based error
            error = np.linalg.norm(b - A @ x_new, ord=np.inf)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            return x_new, k + 1
        else:
            print("Step {} Error {:10.6g}".format(k + 1, error))
        x = x_new

    return x, max_iter


def jacobi_with_precomputed_b(A, b, x0, tol, max_iter=100, opt=0):
    # Extract D, L, U
    d = np.diag(A)
    D = np.diag(d)
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    # Precompute D^-1 and D^-1 * b
    D_inv = np.diag(1 / d)
    c = D_inv @ b  # Precomputed D^-1 * b

    # Initialize variables
    x = x0.copy()
    for k in range(max_iter):
        # Compute the next iteration
        x_new = -D_inv @ (L + U) @ x + c

        # Error calculation based on OPT parameter
        if opt == 0:  # Increment-based error
            error = np.linalg.norm(x_new - x)
        elif opt == 1:  # Residual-based error
            error = np.linalg.norm(b - A @ x_new)

        # Convergence check
        if error < tol:
            return x_new, k + 1
        else:
            print("Step {} Error {:10.6g}".format(k + 1, error))

        x = x_new  # Update for the next iteration

    raise ValueError(
        "Method did not converge within the maximum number of iterations")
