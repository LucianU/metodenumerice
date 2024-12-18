import numpy as np
import time


def sor_method(A, b, x0, omega=1.0, tolerance=1e-6, max_iterations=100, opt=0):
    t = time.time()
    x = x0.copy()

    iter1 = 0
    # Iterate
    for _ in range(max_iterations):
        iter1 = iter1 + 1
        print("The solution vector in iteration", iter1, "is:", x)
        x_old = x.copy()

        # Loop over rows
        rows = A.shape[0]
        for i in range(rows):
            x[i] = x[i] * (1 - omega) + (omega / A[i, i]) * (
                b[i] - np.dot(A[i, :i], x[:i])
                     - np.dot(A[i, (i + 1):], x_old[(i + 1):])
            )

        # Stop condition
        # LnormInf corresponds to the absolute value of the greatest element of the vector.
        if opt == 0:
            LnormInf = max(abs((x - x_old))) / max(abs(x_old))
            print("The L infinity norm in iteration", iter1, "is:", LnormInf)
            if LnormInf < tolerance:
                break
        elif opt == 1:
            error = np.linalg.norm(b - A @ x, ord=np.inf)
            if error < tolerance:
                break
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

    elapsed = time.time() - t
    print("Time elapsed is", elapsed)
    return x


def sor(A, b, x0, omega=1.0, tol=1e-6, max_iter=100, opt=0):
    x = x0.copy()

    for k in range(max_iter):
        x_new = x0.copy()
        for i in range(A.shape[0]):
            sigma = 0
            for j in range(A.shape[1]):
                if j != 1:
                    sigma += A[i, j] * x[j]
            x_new[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        if opt == 0:
            error = np.linalg.norm(x_new - x, ord=np.inf)
        elif opt == 1:
            error = np.linalg.norm(A @ x_new - b, ord=np.inf)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            return x_new, k + 1
        else:
            print("Step {} Error {:10.6g}".format(k + 1, error))
        x = x_new


def sor_numpy(A, b, x0, omega, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()

    for iteration in range(max_iter):
        x_new = x.copy()

        # SOR iteration row by row
        for i in range(n):
            # Compute the sum for elements BEFORE the diagonal (already updated values)
            lower_sum = np.dot(A[i, :i], x_new[:i])

            # Compute the sum for elements AFTER the diagonal (old values)
            upper_sum = np.dot(A[i, i+1:], x[i+1:])

            # Standard Gauss-Seidel value
            x_gs = (b[i] - lower_sum - upper_sum) / A[i, i]

            # Apply relaxation factor
            x_new[i] = (1 - omega) * x[i] + omega * x_gs

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
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

    omega = 1.5  # Relaxation factor (try values between 1 and 2)
    x_sor = sor_numpy(A, b, x0, omega)
    print("Solution:", x_sor)

    A_1 = np.array([[4.0, 1.0, 1.0],
                    [2.0, -9.0, 0.0],
                    [0.0, -8.0, -6.0]])
    b_1 = np.array([6, -7, -14])
    x_1 = sor_method(A_1, b_1, 1, 0,  1e-8, 1000)
    print(x_1)
