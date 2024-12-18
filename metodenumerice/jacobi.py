import numpy as np
from numpy.linalg import norm

from .error import print_error
from .utils import dul_descomp, check_args


def jacobi_iter(A, b, x0, TOL, max_it):
    n = A.shape[0]
    x = x0.copy()

    err = 1.0
    k_f = 0

    while k_f < max_it:
        for i in range(n):
            x_k = x.copy()
            temp_i_row = A[i, :].copy()
            temp_i_row[i] = 0.0
            x_k[i] = (-1) * (1 / A[i][i]) * (temp_i_row @ x_k + b[i])

        err = norm(x_k - x, np.inf) / norm(x_k, np.inf)
        if err < TOL:
            x = x_k
            k_f += 1
            break
        else:
            x = x_k
            k_f += 1
    return x, err, k_f


@check_args
def jacobi(A, b, x0, tol=1e-6, max_iter=100, opt=0):
    x = x0.copy()
    d = np.diag(A)
    D = np.diag(d)
    R = A - D

    D_inv = np.diag(1 / d)

    for k in range(max_iter):
        x_new = (b - R @ x) @ D_inv

        if opt == 0:  # Increment-based error
            error = np.linalg.norm(x_new - x, ord=np.inf)
        elif opt == 1:  # Residual-based error
            error = np.linalg.norm(b - A @ x_new, ord=np.inf)
        else:
            raise ValueError(
                "Invalid OPT value. Use 0 (increment) or 1 (residual).")

        if error < tol:
            return x_new, k + 1
        else:
            print_error(k + 1, error)
        x = x_new

    return x, max_iter


@check_args
def jacobi_with_precomputed_b(A, b, x0, tol, max_iter=100, opt=0):
    d = np.diag(A)
    _, U, L = dul_descomp(A)

    # Precompute D^-1 and D^-1 * b
    D_inv = np.diag(1 / d)
    c = D_inv @ b  # Precomputed D^-1 * b

    # Initialize variables
    x = x0.copy()
    for k in range(max_iter):
        # Compute the next iteration
        x_new = -D_inv @ (L + U) @ x + c

        # Error calculation based on OPT parameter
        if opt == 0:  # Increment-based error
            error = norm(x_new - x)
        else:  # Residual-based error
            error = norm(b - A @ x_new)

        # Convergence check
        if error < tol:
            return x_new, k + 1
        else:
            print("Step {} Error {:10.6g}".format(k + 1, error))

        x = x_new  # Update for the next iteration

    raise ValueError(
        "Method did not converge within the maximum number of iterations")


from typing import List


def jacobi_method(A: List[List[int]], b: List[int], tolerance: float, max_iterations: int) -> List[float]:
    """
    Solves the system of linear equations Ax = b using the Jacobi method.

    Parameters:
    A (List[List[float]]): The coefficient matrix.
    b (List[float]): The right-hand side vector.
    tolerance (float): The convergence tolerance.
    max_iterations (int): The maximum number of iterations to perform.

    Returns:
    List[float]: The approximate solution vector x.
    """

    # Step 1: Get the size of the system (number of rows in A)
    n = len(A)

    # Step 2: Initialize the solution vector x with zeros
    # x_old stores the solution from the previous iteration
    x_old = [0.0 for _ in range(n)]

    # Step 3: Initialize the new solution vector x_new
    # x_new stores the updated solution during the current iteration
    x_new = [0.0 for _ in range(n)]

    # Step 4: Start the iteration process
    for iteration in range(max_iterations):
        # Display the current iteration number
        print(f"Iteration {iteration + 1}:")

        # Step 5: Update each component of x_new based on the Jacobi formula
        for i in range(n):
            # Compute the sum of A[i][j] * x_old[j] for all j except i
            sum_except_i = 0.0
            for j in range(n):
                if j != i:
                    # A[i][j] * x_old[j] is added to the sum
                    sum_except_i += A[i][j] * x_old[j]

            # Compute the new value of x[i]
            # x_new[i] = (b[i] - sum_except_i) / A[i][i]
            x_new[i] = (b[i] - sum_except_i) / A[i][i]

            # Display the computation for this component
            print(f"  x[{i}] = ({b[i]} - {sum_except_i}) / {A[i][i]} = {x_new[i]}")

        # Step 6: Check for convergence
        # Compute the difference between x_new and x_old
        max_difference = max(abs(x_new[i] - x_old[i]) for i in range(n))

        # Display the convergence information
        print(f"  Maximum difference: {max_difference}")

        # If the maximum difference is less than the tolerance, stop iterating
        if max_difference < tolerance:
            print("Converged!")
            return x_new

        # Step 7: Update x_old for the next iteration
        # Copy the values of x_new into x_old
        x_old = x_new[:]

        # Display the updated solution
        print(f"  Updated solution: {x_new}")

    # Step 8: If the method did not converge within the maximum number of iterations
    print("Did not converge within the maximum number of iterations.")
    return x_new


# This doesn't actually exist in the literature, so I wonder which method it is.
# Looks similar to Gauss-Seidel.
def jacobi_backwards(A: List[List[int]], b: List[int], tolerance: float, max_iterations: int) -> List[float]:
    """
    Solves the system of linear equations Ax = b using the backwards Jacobi method.

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
    x_old = [0.0 for _ in range(n)]
    x_new = [0.0 for _ in range(n)]

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")  # Display current iteration number

        for i in range(n):
            # Compute the sum of A[i][j] * x_new[j] for all j < i (backwards updates)
            sum_before_i = 0.0
            for j in range(i):
                sum_before_i += A[i][j] * x_new[j]

            # Compute the sum of A[i][j] * x_old[j] for all j > i
            sum_after_i = 0.0
            for j in range(i + 1, n):
                sum_after_i += A[i][j] * x_old[j]

            # Update x_new[i] using the Jacobi formula
            x_new[i] = (b[i] - sum_before_i - sum_after_i) / A[i][i]

            # Display detailed computation for this element
            print(f"  x[{i}] = ({b[i]} - {sum_before_i} - {sum_after_i}) / {A[i][i]} = {x_new[i]}")

        # Check for convergence
        max_difference = max(abs(x_new[i] - x_old[i]) for i in range(n))
        print(f"  Maximum difference: {max_difference}")

        if max_difference < tolerance:
            print("Converged!")
            return x_new

        # Update x_old for the next iteration
        x_old = x_new[:]
        print(f"  Updated solution: {x_new}")

    print("Did not converge within the maximum number of iterations.")
    return x_new


if __name__ == "__main__":
    # Define the coefficient matrix A
    A = [
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 3]
    ]

    # Define the right-hand side vector b
    b = [15, 10, 10, 10]

    # Define the tolerance for convergence
    tolerance = 1e-6

    # Define the maximum number of iterations
    max_iterations = 100

    # Solve the system using the Jacobi method
    x_jacobi = jacobi_method(A, b, tolerance, max_iterations)
    x_jacobi_backwards = jacobi_backwards(A, b, tolerance, max_iterations)

    print("\nFinal solution: ", x_jacobi)
    print("\nFinal solution: ", x_jacobi_backwards)

