import numpy as np

def sdm(A, b, x0=None, tol=1e-6, maxiter=1000, debug=False):
    """
    Solve A*x = b using the steepest descent algorithm.

    Parameters:
        A (ndarray): Coefficient matrix.
        b (ndarray): Right-hand side vector.
        x (ndarray): Initial guess for the solution.
        tol (float): Tolerance for convergence.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (ndarray): Solution vector.
        iter (int): The number of iterations.
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    x = x0
    iter = 1
    r = b - np.dot(A, x)
    delta = np.dot(r, r)
    delta0 = delta

    while (delta > tol * delta0) and (iter < maxiter):
        q = np.dot(A, r)
        alpha = delta / np.dot(q, r)
        x = x + alpha * r

        if iter % 50 == 0:
            r = b - np.dot(A, x)  # Recalculate r once in a while
        else:
            r = r - alpha * q

        delta = np.dot(r, r)
        iter += 1

        if debug:
            print(f"Residual at step {iter}: {delta}")

    return x, iter

# Example usage
if __name__ == "__main__":
    # Example system A*x = b
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)

    x, iters = sdm(A, b)
    print("Solution:", x)
    print("Iterations:", iters)
