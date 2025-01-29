import numpy as np

def cgne(A, b, x0=None, tol=1e-6, maxiter=1000, debug=False):
    """
    Solve a least squares problem using the Conjugate Gradient method on the Normal Equations (CGNE).

    Parameters:
        A (ndarray): Coefficient matrix.
        b (ndarray): Right-hand side vector.
        x0 (ndarray): Initial guess for the solution.
        tol (float): Convergence tolerance for the residual.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (ndarray): Solution vector.
        iters (int): Number of iterations performed.
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    x = x0
    r = np.dot(A.T, (b - np.dot(A, x)))  # Initial residual (projected into A^T space)
    p = r.copy()
    iters = 0

    for _ in range(maxiter):
        Ap = np.dot(A, p)
        AtAp = np.dot(A.T, Ap)  # Efficient calculation of A^T * (A * p)
        alpha = np.dot(r, r) / np.dot(p, AtAp)
        x = x + alpha * p
        r_new = r - alpha * AtAp

        # Convergence check
        residual_norm = np.linalg.norm(r_new)
        if debug:
            print(f"Residual at step {iters + 1}: {residual_norm}")

        if residual_norm < tol:
            break

        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        iters += 1

    return x, iters

# Example usage
if __name__ == "__main__":
    # Example system A * x ≈ b
    A = np.array([[3, 2], [2, 6]], dtype=float)
    b = np.array([2, -8], dtype=float)

    # Solve using CGNE
    x, iters = cgne(A, b)
    print("Solution:", x)
    print("Iterations:", iters)
