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


def gauss_seidel_solve(A, b, x0, tol=1e-6, max_iter=1000):
    x = x0.copy()

    # Splitting the matrix A
    L_plus_D = np.tril(A)  # Lower triangular + diagonal
    U = np.triu(A, k=1)    # Upper triangular (excluding diagonal)

    for iteration in range(max_iter):
        # Compute the right-hand side: b - U @ x
        rhs = b - np.dot(U, x)

        # Solve for x_new using (L + D) @ x_new = rhs
        x_new = np.linalg.solve(L_plus_D, rhs)

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {iteration+1} iterations.")
            return x_new

        # Update x for the next iteration
        x = x_new.copy()

    print("Max iterations reached without convergence.")
    return x


def gauss_seidel_numpy(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = x0.copy()

    for iteration in range(max_iter):
        x_new = x.copy()

        # Perform the Gauss-Seidel update row by row
        for i in range(n):
            # Calculate the sum of A[i, j] * x_new[j] for j < i (already updated values)
            lower_sum = np.dot(A[i, :i], x_new[:i])

            # Calculate the sum of A[i, j] * x[j] for j > i (old values)
            upper_sum = np.dot(A[i, i+1:], x[i+1:])

            # Update x_new[i] explicitly
            x_new[i] = (b[i] - lower_sum - upper_sum) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {iteration+1} iterations.")
            return x_new

        # Update x for the next iteration
        x = x_new

    print("Max iterations reached without convergence.")
    return x


from typing import List

def gauss_seidel(A: List[List[float]], b: List[float], tolerance: float, max_iterations: int) -> List[float]:
    """
    Solves the system of linear equations Ax = b using the Gauss-Seidel method.

    Parameters:
    A (List[List[float]]): The coefficient matrix.
    b (List[float]): The right-hand side vector.
    tolerance (float): The convergence tolerance.
    max_iterations (int): The maximum number of iterations to perform.

    Returns:
    List[float]: The approximate solution vector x.
    """
    n = len(A)  # Number of equations

    # Initialize the solution vector with zeros
    x = [0.0 for _ in range(n)]

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")

        max_difference = 0.0

        for i in range(n):
            # Compute the sum of A[i][j] * x[j] for all j != i
            sum_except_i = 0.0
            for j in range(n):
                if j != i:
                    sum_except_i += A[i][j] * x[j]

            # Update x[i] using the Gauss-Seidel formula
            new_x_i = (b[i] - sum_except_i) / A[i][i]

            # Compute the difference for convergence checking
            difference = abs(new_x_i - x[i])
            max_difference = max(max_difference, difference)

            # Update x[i] immediately
            print(f"  x[{i}] = ({b[i]} - {sum_except_i}) / {A[i][i]} = {new_x_i}")
            x[i] = new_x_i

        print(f"  Maximum difference: {max_difference}")

        if max_difference < tolerance:
            print("Converged!")
            return x

        print(f"  Updated solution: {x}")

    print("Did not converge within the maximum number of iterations.")
    return x


if __name__ == '__main__':
    A = np.array([[4, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 4]], dtype=float)
    b = np.array([15, 10, 10], dtype=float)
    x0 = np.zeros_like(b)

    x = gauss_seidel_numpy(A, b, x0)
    x_solve = gauss_seidel_solve(A, b, x0)

    print("Solution: ", x)
    print("Solution `linalg.solve`: ", x_solve)

