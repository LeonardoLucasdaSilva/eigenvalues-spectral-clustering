import numpy as np

from utils.lanczos import largest_eigenvalues, tridiag_eigenvector


def lanczos(A, v1, k, m = 100, tol = 1.0e-14):

    n = A.shape[0]
    V = np.zeros((n,m+1))
    V[:,0] = v1/np.linalg.norm(v1)

    a = np.zeros(m)
    b = np.zeros(m)

    w = A @ V[:,0]

    for j in range(m):
        a[j] = w.T @ V[:, j]
        w = w - a[j] * V[:, j]

        b[j] = np.linalg.norm(w)
        if b[j] < tol:
            m = j
            break

        V[:, j + 1] = w / b[j]
        w = A @ V[:, j + 1] - b[j] * V[:, j]

    ritz_vectors = []

    eigs = largest_eigenvalues(a, b, k)
    for lam in eigs:

        y = tridiag_eigenvector(a,b,lam)
        u = V[:, :m] @ y
        ritz_vectors.append(u)


    return eigs, ritz_vectors



