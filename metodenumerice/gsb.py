import numpy as np

from .error import increment, residual
from .utils import check_args


@check_args
def gsb(A, b, x0, tol=1e-6, max_iter=1000, opt=0):
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

            # Update x_new[i]
            x_new[i] = (b[i] - upper_sum - lower_sum) / A[i, i]

        if opt == 0:
            error = increment(x_new, x)
        else:
            error = residual(A, x_new, b)

        # Check for convergence
        if error < tol:
            print(f"Converged in {iteration+1} iterations.")
            return x_new

        # Update x for the next iteration
        x = x_new

    print("Max iterations reached without convergence.")
    return x


from typing import List

def gauss_seidel_backwards(A: List[List[int]], b: List[int], tolerance: float, max_iterations: int) -> List[float]:
    """
    Solves the system of linear equations Ax = b using the backwards Gauss-Seidel method.

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

        for i in range(n - 1, -1, -1):  # Iterate backwards from n-1 to 0
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

    x = gsb(A, b, x0)
    print("Solution: ", x)

    # Define the coefficient matrix A
    A1 = [
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 3]
    ]

    # Define the right-hand side vector b
    b1 = [15, 10, 10, 10]

    # Define the tolerance for convergence
    tolerance = 1e-6

    # Define the maximum number of iterations
    max_iterations = 100

    # Solve the system using the backwards Jacobi method
    x_explicit = gauss_seidel_backwards(A1, b1, tolerance, max_iterations)
    print("Solution x_explicit: ", x_explicit)
