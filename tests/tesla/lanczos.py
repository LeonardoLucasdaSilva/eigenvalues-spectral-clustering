import numpy as np
import time

from algorithms.lanczos import lanczos
from algorithms.retrieve_data import matrix_L_sym
from utils.metrics import relative_error

# Build the L_sym matrix
L_sym = matrix_L_sym(True, 0.01, 'tesla')

r1 = np.linalg.eigvals(L_sym)
r1_sorted = np.sort(r1)[::-1]

# Lanczos dimensions to test
k_values = list(range(10, 210, 10))

print("Lanczos method with QR solver for L_sym matrix:\n")

for k in k_values:
    print(f"Running Lanczos with k = {k}")

    times = []

    # Number of repetitions for averaging timing
    n = 5

    for _ in range(n):
        B = L_sym.copy()
        v1 = np.random.rand(B.shape[0])

        start = time.time()
        eigenvalues, ritz_vectors = lanczos(B, v1, k, 500)
        end = time.time()

        times.append(end - start)

    avg_time = np.mean(times)
    print(f"Average time: {avg_time:.6f} seconds")

print(f"Average processing time with {n} samples: {np.mean(times)} seconds")
relative_error(r1[:200], eigenvalues, 'Lanczos method with QR solver')