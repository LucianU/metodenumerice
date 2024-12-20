import numpy as np

from .error import increment, residual


def sorb(A, b, x0, tol=1e-6, max_iter=1000, opt=0):
    n = len(b)
    x = x0.copy()

    for iteration in range(max_iter):
        x_new = x.copy()

        # Process rows from bottom to top
        for i in range(n-1, -1, -1):  # Reverse order
            # Compute the sum for elements AFTER the diagonal (updated already in this iteration)
            upper_sum = np.dot(A[i, i+1:], x_new[i+1:])

            # Compute the sum for elements BEFORE the diagonal (old values)
            lower_sum = np.dot(A[i, :i], x[:i])

            # Standard Gauss-Seidel value
            x_gs = (b[i] - upper_sum - lower_sum) / A[i, i]

            # Apply relaxation factor
            x_new[i] = (1 - omega) * x[i] + omega * x_gs

        # Check for convergence
        if opt == 0:
            error = increment(x_new, x)
        else:
            error = residual(A, x_new, b)

        if error < tol:
            print(f"Converged in {iteration+1} iterations.")
            return x_new

        # Update x for the next iteration
        x = x_new

    print("Max iterations reached without convergence.")
    return x


if __name__ == '__main__':
    A = np.array([[4, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 4]], dtype=float)
    b = np.array([15, 10, 10], dtype=float)
    x0 = np.zeros_like(b)

    A_1 = np.array([[4.0, 1.0, 1.0],
                    [2.0, -9.0, 0.0],
                    [0.0, -8.0, -6.0]])
    b_1 = np.array([6, -7, -14])

    omega = 1.2  # Relaxation factor (try values between 1 and 2)
    x_sorb = sorb(A, b, x0, omega)
    print ("x_sorb: ", x_sorb)

