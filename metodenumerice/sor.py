from .error import increment, residual


def sor(A, b, x0, omega, tol=1e-8, max_iter=1000, opt=0):
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        # SOR iteration row by row
        for i in range(n):
            # Compute the sum for elements BEFORE the diagonal (already updated values)
            L_sum = A[i, :i] @ x_new[:i]

            # Compute the sum for elements AFTER the diagonal (old values)
            U_sum = A[i, i+1:] @ x[i+1:]

            # Standard Gauss-Seidel value
            x_gs = (b[i] - L_sum - U_sum) / A[i, i]

            # Apply relaxation factor
            x_new[i] = (1 - omega) * x[i] + omega * x_gs

        if opt == 0:
            error = increment(x_new, x)
        elif opt == 1:
            error = residual(A, x_new, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            print(f"Converged in {k+1} iterations.")
            return x_new, k+1

        # Update x for the next iteration
        x = x_new
    else:
        print("Max iterations reached without convergence.")
    return x, max_iter

