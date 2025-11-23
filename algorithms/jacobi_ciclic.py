import numpy as np

from utils.givens import givens_row, givens_column
from utils.jacobi import jacobi_args

def jacobi_cyclic(A, maxiter = 10000, tol = 1e-12):

    n = A.shape[0]
    V = np.eye(n)
    prev_err = -1
    for k in range(maxiter):

        for i in range(n-1):
            for j in range(i+1, n):
                c,s = jacobi_args(A[i,i], A[i,j], A[j,j])
                A[[i, j], :] = givens_row(A[[i, j], :], c, s)
                A[:, [i,j]] = givens_column(A[:, [i,j]], c, s)
                V[:, [i,j]] = givens_column(V[:, [i,j]], c, s)

        err = np.max(np.abs(np.tril(A, -1)))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

        print(f"k = {k}, err = {err}")
    return A,V
