import numpy as np
import matplotlib.pyplot as plt


def jor_corrected(A, b, x0, omega, tol=1e-8, max_iter=1000):
    n = len(b)
    x = x0.copy()
    residuals = []

    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i, i]
        error = np.linalg.norm(x_new - x, ord=np.inf)
        residuals.append(error)
        if error < tol:
            print(f"Converged in {k + 1} iterations.")
            break
        x = x_new
    else:
        print("Did not converge.")
    return x, residuals


def jor_numpy_wikipedia(A, b, x0, omega, tol=1e-8, max_iter=1000):
    D_inv = np.diag(1 / np.diag(A))  # D^(-1)
    L_U = A - np.diag(np.diag(A))   # L + U
    x = x0.copy()
    residuals = []

    for k in range(max_iter):
        x_new = omega * D_inv @ (b - L_U @ x) + (1 - omega) * x
        error = np.linalg.norm(x_new - x, ord=np.inf)
        residuals.append(error)

        if error < tol:
            print(f"Converged in {k + 1} iterations.")
            break
        x = x_new.copy()
    else:
        print("Didn't converge.")
    return x, residuals


def jor(A, b, x0, omega, tol=1e-6, max_iter=100):
    """Jacobi Over-Relaxation method."""
    x = x0.copy()
    D_inv = np.diag(1 / np.diag(A))  # Diagonal inverse for Jacobi iteration
    residuals = []

    for _ in range(max_iter):
        x_new = x + omega * (D_inv @ (b - A @ x) - x)
        error = np.linalg.norm(b - A @ x_new, ord=2) / np.linalg.norm(b, ord=2)
        residuals.append(error)
        if error < tol:
            break
        x = x_new
    return x, residuals


def sor(A, b, x0, omega, tol=1e-6, max_iter=100):
    """Successive Over-Relaxation method."""
    n = len(b)
    x = x0.copy()
    residuals = []

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            lower_sum = np.dot(A[i, :i], x_new[:i])
            upper_sum = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = x[i] + omega * ((b[i] - lower_sum - upper_sum) / A[i, i] - x[i])
        error = np.linalg.norm(b - A @ x_new, ord=2) / np.linalg.norm(b, ord=2)
        residuals.append(error)
        if error < tol:
            break
        x = x_new
    return x, residuals


# Test case
A = np.array([[4, -1, 0],
              [-1, 4, -1],
              [0, -1, 4]], dtype=float)
b = np.array([15, 10, 10], dtype=float)
x0 = np.zeros_like(b)
joromega = 0.6
omega = 1.2

# Run JOR and SOR
x_jor, res_jor = jor_numpy_wikipedia(A, b, x0, joromega)
#x_jor, res_jor = jor(A, b, x0, joromega)
x_sor, res_sor = sor(A, b, x0, omega)

# Plot residual errors
plt.figure(figsize=(8, 6))
plt.semilogy(range(len(res_jor)), res_jor, label='JOR', marker='o')
plt.semilogy(range(len(res_sor)), res_sor, label='SOR', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Relative Residual Error (log scale)')
plt.title('Comparison of JOR and SOR Error Evolution')
plt.legend()
plt.grid()
plt.show()

#print("JOR corr: ", x_jor_cor)
print("JOR Solution:", x_jor)
print("SOR Solution:", x_sor)
