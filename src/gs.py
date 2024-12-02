import numpy as np


def gs(a, b, TOL, max_it):
    n = len(a)
    x = np.zeros(n)

    for iteration in range(max_it):
        previous_x = x.copy()

        for j in range(n):
            d = b[j]
            for i in range(n):
                if j != i:
                    d -= a[j][i] * x[i]
            x[j] = d / a[j][j]
        print("Iteration", iteration + 1, ":", x)

        # Check for convergence
        if all(abs(x[i] - previous_x[i]) < TOL for i in range(n)):
            print("Converged after", iteration + 1, "iterations.")
            break
    else:
        print("Maximum iterations reached without convergence.")

    return x
