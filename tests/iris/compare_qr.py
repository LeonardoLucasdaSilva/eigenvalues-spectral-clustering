import numpy as np
from algorithms.qr_iteration import qr_iteration_householder_shift, qr_iteration_householder_shift_deflation, \
    qr_iteration_householder_deflation
from algorithms.retrieve_data import matrix_L_sym
from utils.metrics import relative_error
import time
import matplotlib.pyplot as plt

# Build the L_sym matrix
L_sym = matrix_L_sym(True, 0.01, 'iris')

# Retrieve eigenvalues using numpy for comparison
r1 = np.linalg.eigvals(L_sym)
times = []

# Number of samples to average time
n = 1

# print("QR iteration with shift for L_sym matrix:")
# for k in range(n):
#
#     start = time.time()
#
#     H,Q = qr_iteration_householder_shift(L_sym, tol = 5e-8)
#
#     end = time.time()
#
#     # Retrieve eigenvalues from diagonal
#     r2 = np.diag(H)
#
#     times.append(end - start)
#
# print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
# error_shift, count_error_shift = relative_error(r1,r2,title = "Shifted QR")
#
# print("QR iteration with deflation for L_sym matrix:")
# for k in range(n):
#
#     start = time.time()
#
#     H,Q = qr_iteration_householder_deflation(L_sym, tol = 5e-8)
#
#     end = time.time()
#
#     # Retrieve eigenvalues from diagonal
#     r3 = np.diag(H)
#
#     times.append(end - start)
#
# print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
# error_deflation,count_error_deflation = relative_error(r1,r3, title="Deflated QR")

print("QR iteration with shift and deflation for L_sym matrix:")
for k in range(n):

    start = time.time()

    H,Q = qr_iteration_householder_shift_deflation(L_sym, tol = 5e-8)

    end = time.time()

    # Retrieve eigenvalues from diagonal
    r4 = np.diag(H)

    times.append(end - start)

print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
error_shift_deflation, count_error_shift_deflation = relative_error(r1,r4, title = "Shifted-deflated QR")

plt.plot(error_shift, marker='o', label='Shifted QR')
plt.plot(error_deflation, marker='s', label='Deflation QR')
plt.plot(error_shift_deflation, marker='^', label='Shifted + Deflation QR')

plt.yscale("log")   # <<--- Logarithmic scale
plt.xlabel("Component Index")
plt.ylabel("Relative Error (log scale)")
plt.title("Comparison of QR Variants – Relative Errors")
plt.grid(True, which="both", ls='--')
plt.legend()
plt.tight_layout()
plt.show()



