import numpy as np

from algorithms.power_method import shifted_inverse_power_method
from algorithms.qr_iteration import qr_iteration_householder_shift_deflation
from algorithms.retrieve_data import matrix_L_sym
from joblib import Parallel, delayed
import time

from utils.metrics import relative_error

# Build the L_rw matrix
L_sym = matrix_L_sym(True, 0.01, 'tesla')

# Retrieve eigenvalues using numpy for comparison
r1 = np.linalg.eigvals(L_sym)
times = []

# Number of samples to average time
n = 1

n_eigs = 100

print("Power method with QR for L_sym matrix:")
for k in range(n):
    start = time.time()

    H, Q = qr_iteration_householder_shift_deflation(L_sym, max_iter=50, tol=5e-3)
    diag = np.diag(H)

    results = Parallel(n_jobs=-1)(
        delayed(shifted_inverse_power_method)(L_sym, Q[i, :], diag[i])
        for i in range(len(diag[:n_eigs]))
    )

    end = time.time()

    times.append(end - start)

    r3 = [float(val) for val, vec in results]

print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
relative_error(r1[:n_eigs], r3, "Power method with QR for L_sym matrix")