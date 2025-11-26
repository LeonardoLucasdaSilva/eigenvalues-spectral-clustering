import numpy as np
from algorithms.qr_iteration import qr_iteration_householder_deflation
from algorithms.retrieve_data import matrix_L_sym
from utils.metrics import relative_error
import time

# Build the L_sym matrix
L_sym = matrix_L_sym(True, 0.01, 'iris')

# Retrieve eigenvalues using numpy for comparison
r1 = np.linalg.eigvals(L_sym)
times = []

# Number of samples to average time
n = 5

print("QR iteration with deflation for L_sym matrix:")
for k in range(n):

    start = time.time()

    H,Q = qr_iteration_householder_deflation(L_sym, tol = 1e-8)

    end = time.time()

    # Retrieve eigenvalues from diagonal
    r2 = np.diag(H)

    times.append(end - start)

print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
error = relative_error(r1,r2, title="Deflated QR")