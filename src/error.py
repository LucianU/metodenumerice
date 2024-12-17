import numpy as np


def step(x_new, x_old, norm=2):
    """
    Compute the error as the difference between current and previous approximations.

    Parameters:
        x_new (ndarray): Current iteration vector.
        x_old (ndarray): Previous iteration vector.
        norm (int): Order of the norm (e.g., 1, 2, np.inf).

    Returns:
        float: The error between two iterations.
    """
    step = np.linalg.norm(x_new - x_old, ord=norm)
    return step


def residual(A, x, b, norm=2):
    """
    Compute the residual error as the norm of (A @ x - b).

    Parameters:
        A (ndarray): Coefficient matrix.
        x (ndarray): Current iteration vector.
        b (ndarray): Right-hand side vector.
        norm (int): Order of the norm (e.g., 1, 2, np.inf).

    Returns:
        float: The residual error.
    """
    residual = np.linalg.norm(A @ x - b, ord=norm)
    return residual


def print_error(step, error):
    print("Step {} Error {:10.6g}".format(step, error))
