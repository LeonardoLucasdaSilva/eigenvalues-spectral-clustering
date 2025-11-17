import numpy as np
from algorithms.qr_iteration import qr_iteration_householder_h

def arnoldi(A, v1, kmax, tol=1e-12):
    n = A.shape[0]
    V = np.zeros((n, kmax+1))
    H = np.zeros((kmax+1, kmax))

    V[:, 0] = v1 / np.linalg.norm(v1)

    for i in range(kmax):
        avj = A @ V[:, i]

        # Compute all H[j,i]
        for j in range(i+1):
            H[j, i] = np.dot(V[:, j], avj)

        # Orthogonalize
        vhat = avj.copy()
        for j in range(i+1):
            vhat -= H[j, i] * V[:, j]

        H[i+1, i] = np.linalg.norm(vhat)

        if H[i+1, i] < tol:
            T,Qe = qr_iteration_householder_h(H[:i+1, :i+1])
            Q = V[:, :i+1] @ Qe

            return T,Q

        V[:, i+1] = vhat / H[i+1, i]

    return H[:kmax, :kmax], np.eye(n)




