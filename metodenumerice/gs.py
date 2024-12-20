from .error import increment, residual

from .sor import sor


def gs(A, b, x0, tol=1e-6, max_iter=1000, opt=0):
    return sor(A, b, x0, 1.0, tol, max_iter, opt)


def gs_base(A, b, x0, tol=1e-6, max_iter=1000, opt=0):
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        # Perform the Gauss-Seidel update row by row
        for i in range(n):
            # Calculate the sum of A[i, j] * x_new[j] for j < i (already updated values)
            L_sum = A[i, :i] @ x_new[:i]

            # Calculate the sum of A[i, j] * x[j] for j > i (old values)
            U_sum = A[i, i+1:] @ x[i+1:]

            # Update x_new[i] explicitly
            x_new[i] = (b[i] - L_sum - U_sum) / A[i, i]

        if opt == 0:
            error = increment(x_new, x)
        elif opt == 1:
            error = residual(A, x_new, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            print(f"GS: Converged in {k + 1} iterations.")
            return x_new, k + 1

        # Update x for the next iteration
        x = x_new
    else:
        print("GS: Didn't converge.")

    return x, max_iter

