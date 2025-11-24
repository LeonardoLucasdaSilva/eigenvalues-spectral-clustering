import numpy as np
from algorithms.qr_iteration import qr_iteration_householder_deflation
from algorithms.retrieve_data import matrix_L_rw
from utils.metrics import relative_error
import time

# Build the L_rw matrix
L_rw = matrix_L_rw(True, 0.01, 'iris')

# Retrieve eigenvalues using numpy for comparison
r1 = np.linalg.eigvals(L_rw)
times = []

# Number of samples to average time
n = 1

print("QR iteration with deflation for L_rw matrix:")
for k in range(n):
    start = time.time()

    H, Q = qr_iteration_householder_deflation(L_rw, tol = 1e-8)

    end = time.time()

    # Retrieve eigenvalues from diagonal
    r2 = np.diag(H)

    times.append(end - start)

print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
relative_error(r1, r2)