from .error import increment, residual


def gsb(A, b, x0, tol=1e-8, max_iter=1000, opt=0):
    n = len(b)
    x = x0.copy()

    for k in range(max_iter):
        x_new = x.copy()

        # Process rows from bottom to top
        for i in range(n-1, -1, -1):  # Reverse order
            # Compute the sum for elements AFTER the diagonal
            U_sum = A[i, i+1:] @ x[i+1:]

            # Compute the sum for elements BEFORE the diagonal
            L_sum = A[i, :i] @ x_new[:i]

            # Update x_new[i]
            x_new[i] = (b[i] - U_sum - L_sum) / A[i, i]

        if opt == 0:
            error = increment(x_new, x)
        elif opt == 1:
            error = residual(A, x_new, b)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        # Check for convergence
        if error < tol:
            print(f"Converged in {k+1} iterations.")
            return x_new, k+1

        # Update x for the next iteration
        x = x_new

    else:
        print("Max iterations reached without convergence.")
    return x, max_iter

