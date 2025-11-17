import numpy as np

from utils.givens import givens_row, givens_column
from utils.jacobi import jacobi_args

def jacobi_cyclic(A, maxiter = 15, tol = 1e-12):

    n = A.shape[0]
    V = np.eye(n)

    for k in range(maxiter):

        for i in range(n-1):
            for j in range(i+1, n):
                c,s = jacobi_args(A[i,i], A[i,j], A[j,j])
                A[[i, j], :] = givens_row(A[[i, j], :], c, s)
                A[:, [i,j]] = givens_column(A[:, [i,j]], c, s)
                V[:, [i,j]] = givens_column(V[:, [i,j]], c, s)

        if np.linalg.norm(A-np.diag(A), ord='fro')<=tol:
            break

    return A,V
