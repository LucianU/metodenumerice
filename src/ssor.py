import numpy as np
from scipy.linalg import norm as norm
from scipy.sparse import diags




"""The program is to split the matrix into D-diagonal; L: strictly lower matrix; U strictly upper matrix
    satisfying: A = D - L - U  """


def splitMat(A):
    n, m = A.shape

    if (n == m):
        diagval = np.diag(A)
        D = diags(diagval, 0).toarray()
        L = (-1) * np.tril(A, -1)
        U = (-1) * np.triu(A, 1)
    else:
        raise ValueError("A needs to be a square matrix")
    return (L, D, U)


"""Preconditioned Matrix for symmetric successive over-relaxation (SSOR): """


def P_SSOR(A,w):
    L,D,U = splitMat(A)
    P = 2/(2-w) * (1/w*D+L)*np.linalg.inv(D)*(1/w*D+L).T
    return P


"""GMRES_SSOR using right preconditioning P:
A - matrix of linear system Ax = b
x0 - initial guess
tol - tolerance
maxit - maximum iteration """


def myGMRES_SSOR(A, x0, b, tol, maxit):
    matrixSize = A.shape[0]
    x0 = np.zeros(matrixSize)
    e = np.zeros((maxit + 1, 1))
    rr = 1
    rstart = 2
    X = x0
    w = 1.9  ## in ssor
    P = P_SSOR(A, w)  ### preconditioned matrix
    ### Starting the GMRES ####
    for rs in range(0, rstart + 1):
        ### first check the residual:
        if rr < tol:
            break
        else:
            r0 = (b - A.dot(x0))
            rho = norm(r0)
            e[0] = rho
            H = np.zeros((maxit + 1, maxit))
            Qcol = np.zeros((matrixSize, maxit+1))
            Qcol[:, 0:1] = r0 / rho
        for k in range(1, maxit + 1):
            ### Arnodi procedure ##
            Qcol[:, k] = np.matmul(np.matmul(A, P), Qcol[:, k - 1])  ### This step applies P here:
            for j in range(0, k):
                H[j, k - 1] = np.dot(np.transpose(Qcol[:, k]), Qcol[:, j])
                Qcol[:, k] = Qcol[:, k] - (np.dot(H[j, k - 1], Qcol[:, j]))

            H[k, k - 1] = norm(Qcol[:, k])
            Qcol[:, k] = Qcol[:, k] / H[k, k - 1]

            ###  QR decomposition step ###
            n = k
            Q = np.zeros((n + 1, n))
            R = np.zeros((n, n))
            R[0, 0] = norm(H[0:n + 2, 0])
            Q[:, 0] = H[0:n + 1, 0] / R[0, 0]
            for j in range(0, n + 1):
                t = H[0:n + 1, j - 1]
                for i in range(0, j - 1):
                    R[i, j - 1] = np.dot(Q[:, i], t)
                    t = t - np.dot(R[i, j - 1], Q[:, i])
                R[j - 1, j - 1] = norm(t)
                Q[:, j - 1] = t / R[j - 1, j - 1]

            g = np.dot(np.transpose(Q), e[0:k + 1])
            Y = np.dot(np.linalg.inv(R), g)

            Res = e[0:n] - np.dot(H[0:n, 0:n], Y[0:n])
            rr = norm(Res)

            #### second check on the residual ###
            if rr < tol:
                break

                #### Updating the solution with the preconditioned matrix ####
        X = X + np.matmul(np.matmul(P, Qcol[:, 0:k]), Y)  ### This steps applies P here:
    return X


A_1 = np.array([[4.0, 1.0, 1.0], [2.0, -9.0, 0.0], [0.0, -8.0, -6.0]])
b_1 = np.array([6, -7, -14])
x0 = np.zeros(3)
x_1 = myGMRES_SSOR(A_1,x0,b_1,1e-8, 1000)

print(x_1)

