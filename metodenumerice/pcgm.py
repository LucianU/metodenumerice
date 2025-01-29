import numpy as np


def pcgm(A, b, x0=None, M=None, tol=1e-6, maxiter=1000):
    """
    Solve the system A*x = b using the Preconditioned Conjugate Gradient method.

    Parameters:
        A (ndarray): Positive-definite symmetric matrix.
        b (ndarray): Right-hand side vector.
        x0 (ndarray): Initial guess for the solution.
        M (ndarray or None): Preconditioner matrix. If None, no preconditioner is used.
        tol (float): Convergence tolerance for the residual.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (ndarray): Solution vector.
        iters (int): Number of iterations performed.
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    x = x0
    r = b - np.dot(A, x)

    # Apply the preconditioner: solve M*z = r
    if M is None:
        z = r  # No preconditioner, M is identity
    else:
        z = np.linalg.solve(M, r)

    p = z
    rz_old = np.dot(r, z)
    iters = 0

    for _ in range(maxiter):
        Ap = np.dot(A, p)
        alpha = rz_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        # Check for convergence
        if np.linalg.norm(r) < tol:
            break

        # Update z using the preconditioner
        if M is None:
            z = r
        else:
            z = np.linalg.solve(M, r)

        rz_new = np.dot(r, z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        iters += 1

    return x, iters

# Example usage
if __name__ == "__main__":
    # Example system A*x = b
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    M = np.diag(np.diag(A))  # Jacobi preconditioner (diagonal of A)

    x, iters = pcgm(A, b, M=M)
    print("Solution:", x)
    print("Iterations:", iters)
