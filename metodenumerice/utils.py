import numpy as np


def dul_descomp(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    return D, U, L


def rho(A):
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    G = np.linalg.inv(D) @ L_plus_U
    return spectral_radius(G)


def spectral_radius(A):
    eigenvalues = np.linalg.eigvals(A)
    return max(abs(eigenvalues))


def is_diag_dom(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off_diag:
            return False
    return True


def is_symm(A):
    return np.allclose(A, A.T, atol=1e-8)


def is_pd(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

