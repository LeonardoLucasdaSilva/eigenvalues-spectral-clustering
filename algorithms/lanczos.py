import numpy as np
from scipy.linalg import eigh_tridiagonal

def lanczos(A, v1, k, m, tol=1.0e-14):
    n = A.shape[0]
    V = np.zeros((n, m+1), dtype=float)
    V[:, 0] = v1 / np.linalg.norm(v1)

    a = np.zeros(m, dtype=float)
    b = np.zeros(m-1, dtype=float)

    w = A @ V[:, 0]

    final_m = m
    for j in range(m):
        a[j] = V[:, j] @ w
        w = w - a[j] * V[:, j]

        if j > 0:
            w = w - b[j-1] * V[:, j-1]

        # Reorthogonalization
        for i in range(j+1):
            w -= (V[:, i] @ w) * V[:, i]

        if j < m-1:
            b[j] = np.linalg.norm(w)
            if b[j] < tol:
                final_m = j + 1
                break
            V[:, j+1] = w / b[j]
            w = A @ V[:, j+1]

    a = a[:final_m]
    b = b[:final_m-1]

    eigenvalues, Qtri = eigh_tridiagonal(a, b)

    # Select largest k eigenpairs
    idx = np.argsort(eigenvalues)[-k:][::-1]
    eigs = eigenvalues[idx]

    # Build Ritz vectors: u = V_m @ y
    ritz_vectors = [V[:, :final_m] @ Qtri[:, i] for i in idx]

    return eigs, ritz_vectors




