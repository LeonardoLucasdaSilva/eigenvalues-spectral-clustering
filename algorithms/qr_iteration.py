import numpy as np

from utils.householder import hessenberg_matrix
from utils.givens import givens_args, givens_row, givens_column

def qr_iteration_householder(A, max_iter=100, tol=1e-10):

    n = A.shape[0]
    H,Q = hessenberg_matrix(A)

    c = np.zeros(n)
    s = np.zeros(n)

    for k in range(max_iter):

        # H = G^T * H
        for i in range(n-1):

            c[i],s[i] = givens_args(H[i,i],H[i+1,i])
            H[i:i+2,i:n] = givens_row(H[i:i+2,i:n],c[i],s[i])
            Q[i:i + 2, :] = givens_row(Q[i:i + 2, :], c[i], s[i])

        # H = H * G
        for i in range(n-1):

            H[:i+2,i:i+2] = givens_column(H[:i+2,i:i+2],c[i],s[i])

        if np.max(np.abs(np.tril(H, -1))) < tol:
            break

    return H,Q

def qr_iteration_householder_h(H, max_iter=100, tol=1e-10):

    n = H.shape[0]
    Q = np.eye(n)
    H = np.copy(H)

    c = np.zeros(n)
    s = np.zeros(n)

    for k in range(max_iter):

        # H = G^T * H
        for i in range(n-1):

            c[i],s[i] = givens_args(H[i,i],H[i+1,i])
            H[i:i+2,i:n] = givens_row(H[i:i+2,i:n],c[i],s[i])

        # H = H * G
        for i in range(n-1):

            H[:i+2,i:i+2] = givens_column(H[:i+2,i:i+2],c[i],s[i])
            Q[:, i:i+2] = givens_column(Q[:, i:i+2], c[i], s[i])

        if np.max(np.abs(np.tril(H, -1))) < tol:
            break

    return H,Q

# CLASS NOTES
def qr_iteration(A, Q=None, maxiter=100, tol=1e-5):
  Q = np.eye(A.shape[0]) if Q==None else Q
  T0 = Q.T @ A @ Q
  info = -1
  for k in range(maxiter):
    Q, R = np.linalg.qr(T0)
    T = R @ Q
    if (np.linalg.norm(T-T0) < tol):
      info = 0
      break
    T0 = T
  return T, info



