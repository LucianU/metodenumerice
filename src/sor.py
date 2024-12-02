import numpy as np
import time
import numpy.linalg as nla


def sor_method(A, b, omega, initial_guess, tolerance, max_iterations):
    t = time.time()
    x = np.zeros_like(b, dtype=np.double)

    iter1 = 0
    # Iterate
    for k in range(max_iterations):
        iter1 = iter1 + 1
        print("The solution vector in iteration", iter1, "is:", x)
        x_old = x.copy()

        # Loop over rows
        for i in range(A.shape[0]):
            x[i] = x[i] * (1 - omega) + (omega / A[i, i]) * (
                b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, (i + 1):], x_old[(i + 1):]))

            # Stop condition
        # LnormInf corresponds to the absolute value of the greatest element of the vector.

        LnormInf = max(abs((x - x_old))) / max(abs(x_old))
        print("The L infinity norm in iteration", iter1, "is:", LnormInf)
        if LnormInf < tolerance:
            break
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


A_1 = np.array([[4.0, 1.0, 1.0], [2.0, -9.0, 0.0], [0.0, -8.0, -6.0]])
b_1 = np.array([6, -7, -14])
x_1 = sor_method(A_1, b_1, 1, 0,  1e-8, 1000)
print(x_1)
