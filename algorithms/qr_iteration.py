import numpy as np

from utils.householder import hessenberg_matrix
from utils.givens import givens_args, givens_row, givens_column, givens_cs, givens_column_general, givens_row_general
import time
from utils.shifts import wilkinson_shift


def qr_iteration_householder_shift(A, max_iter=10000, tol=1e-10):

    n = A.shape[0]
    H,Q = hessenberg_matrix(A)

    c = np.zeros(n)
    s = np.zeros(n)

    for k in range(max_iter):

        a = H[-2, -2]
        b = H[-2, -1]
        c2 = H[-1, -1]

        d = (a - c2) / 2
        mu = c2 - (np.sign(d) * b ** 2) / (abs(d) + np.sqrt(d ** 2 + b ** 2))

        # apply shift
        H -= mu * np.eye(n)

        # H = G^T * H
        for i in range(n-1):

            c[i],s[i] = givens_args(H[i,i],H[i+1,i])
            H[i:i+2,i:n] = givens_row(H[i:i+2,i:n],c[i],s[i])
            Q[i:i + 2, :] = givens_row(Q[i:i + 2, :], c[i], s[i])

        # H = H * G
        for i in range(n-1):

            H[:i+2,i:i+2] = givens_column(H[:i+2,i:i+2],c[i],s[i])

        # undo shift
        H += mu * np.eye(n)

        error = np.max(np.abs(np.tril(H, -1)))
        print(f'k = {k}, shift = {mu}, error = {error}')

        if error < tol:
            break

    return H,Q

def qr_iteration_householder_shift_deflation(A, max_iter=10000, tol=1e-12):
    n = A.shape[0]

    start = time.time()
    H, Q = hessenberg_matrix(A)
    end = time.time()

    print(f"Hessenberg time: {end-start} seconds")

    # keep m across iterations
    m = n

    for k in range(max_iter):

        # Deflation
        while m > 1 and abs(H[m-1, m-2]) < tol * (abs(H[m-2,m-2]) + abs(H[m-1,m-1])):
            H[m-1, m-2] = 0
            m -= 1

        # If the matrix is fully reduced end loop
        if m <= 1:
            break

        # If we have just a 2x2 block stop
        if m == 2:
            break

        H_active = H[:m, :m]

        # Deals with small values for shift choosing
        if abs(H_active[-1, -1]) < 1e-12:
            mu = 0
        else:
            mu = wilkinson_shift(H_active[-1, -1], H_active[-2, -2], H_active[-2, -1])

        # apply shift
        H_active -= mu * np.eye(m)

        # Givens rotation arrays sized to m
        c = np.zeros(m)
        s = np.zeros(m)

        # Left QR step
        for i in range(m - 1):
            c[i], s[i] = givens_args(H_active[i, i], H_active[i + 1, i])
            H_active[i:i + 2, i:m] = givens_row(H_active[i:i + 2, i:m], c[i], s[i])
            Q[i:i + 2, :] = givens_row(Q[i:i + 2, :], c[i], s[i])

        # Right QR step
        for i in range(m - 1):
            H_active[:i + 2, i:i + 2] = givens_column(H_active[:i + 2, i:i + 2], c[i], s[i])

        # undo shift
        H_active += mu * np.eye(m)
        H[:m, :m] = H_active

        err = np.max(np.abs(np.tril(H, -1)))
        #print(f"k={k}, m={m}, shift={mu:.3e}, error={err:.3e}")

        if err < tol:
            break

    return H, Q

def qr_iteration_householder_deflation(A, max_iter=10000, tol=1e-12):
    n = A.shape[0]
    H, Q = hessenberg_matrix(A)

    # keep m across iterations
    m = n

    for k in range(max_iter):

        # Deflation
        while m > 1 and abs(H[m-1, m-2]) < tol * (abs(H[m-2,m-2]) + abs(H[m-1,m-1])):
            H[m-1, m-2] = 0
            m -= 1

        # If the matrix is fully reduced end loop
        if m <= 1:
            break

        # If we have just a 2x2 block stop
        if m == 2:
            break

        H_active = H[:m, :m]

        c = np.zeros(m)
        s = np.zeros(m)

        # Left QR step
        for i in range(m - 1):
            c[i], s[i] = givens_args(H_active[i, i], H_active[i + 1, i])
            H_active[i:i + 2, i:m] = givens_row(H_active[i:i + 2, i:m], c[i], s[i])
            Q[i:i + 2, :] = givens_row(Q[i:i + 2, :], c[i], s[i])

        # Right QR step
        for i in range(m - 1):
            H_active[:i + 2, i:i + 2] = givens_column(H_active[:i + 2, i:i + 2], c[i], s[i])

        err = np.max(np.abs(np.tril(H, -1)))

        if err < tol:
            break

    return H, Q

def qr_iteration_householder_h(H, max_iter=10000, tol=1e-8):

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


def qr_iteration_tridiagonal(alpha, beta, tol=1e-14, maxiter=10000):
    return ""


# CLASS NOTES
def qr_iteration(A, Q=None, maxiter=10000, tol=1e-5):
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



