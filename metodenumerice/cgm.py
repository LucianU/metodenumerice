import numpy as np

def cgm(A, b, x0=None, tol=1e-8, maxiter=1000):
    """
    Solve A*x = b using the Conjugate Gradient (CG) method.

    Parameters:
        A (ndarray): Symmetric, positive-definite coefficient matrix.
        b (ndarray): Right-hand side vector.
        x0 (ndarray): Initial guess for the solution.
        tol (float): Convergence tolerance.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (ndarray): Solution vector.
        iters (int): Number of iterations performed.
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    x = x0
    r = b - np.dot(A, x)
    p = r.copy()
    rs_old = np.dot(r, r)
    iters = 0

    for _ in range(min(maxiter, len(b))):  # Use maxiter instead of just len(b)
        Ap = np.dot(A, p)
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:  # Convergence check
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        iters += 1

    return x, iters

# Example usage
if __name__ == "__main__":
    # Example system A*x = b
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)

    x, iters = cgm(A, b)
    print("Solution:", x)
    print("Iterations:", iters)

