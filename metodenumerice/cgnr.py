import numpy as np

def cgnr(A, b, x0=None, tol=1e-6, maxiter=1000, debug=False):
    """
    Solve the least squares problem A*x ≈ b using the Conjugate Gradient Normal Residual (CGNR) method.

    Parameters:
        A (ndarray): Coefficient matrix.
        b (ndarray): Right-hand side vector.
        x0 (ndarray): Initial guess for the solution.
        tol (float): Convergence tolerance for the residual norm.
        maxiter (int): Maximum number of iterations.

    Returns:
        x (ndarray): Solution vector.
        iters (int): Number of iterations performed.
    """
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    x = x0
    r = b - np.dot(A, x)  # Residual in data space
    z = np.dot(A.T, r)    # Projected residual in solution space
    p = z.copy()
    iters = 0

    for _ in range(maxiter):
        Ap = np.dot(A, p)
        alpha = np.dot(z, z) / np.dot(Ap, Ap)  # Step size
        x += alpha * p                         # Update solution
        r -= alpha * Ap                        # Update residual

        z_new = np.dot(A.T, r)                 # Project new residual
        residual_norm = np.linalg.norm(r)      # Residual norm

        if debug:
            print(f"Residual at step {iters + 1}: {residual_norm}")

        # Check for convergence
        if residual_norm < tol:
            break

        beta = np.dot(z_new, z_new) / np.dot(z, z)
        p = z_new + beta * p  # Update search direction
        z = z_new
        iters += 1

    return x, iters

# Example usage
if __name__ == "__main__":
    # Example system A*x ≈ b
    A = np.array([[3, 2], [2, 6]], dtype=float)
    b = np.array([2, -8], dtype=float)

    # Solve using CGNR
    x, iters = cgnr(A, b)
    print("Solution:", x)
    print("Iterations:", iters)
