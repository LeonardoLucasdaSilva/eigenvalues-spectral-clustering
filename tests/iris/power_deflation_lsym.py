import numpy as np
import time
import matplotlib.pyplot as plt

from algorithms.power_method import power_method
from algorithms.retrieve_data import matrix_L_sym
from utils.metrics import relative_error

# Build matrix
L_sym = matrix_L_sym(True, 0.01, 'iris')

# Ground truth eigenvalues
r1 = np.linalg.eigvals(L_sym)

# Number of samples per run
n = 5

# Range of k values
k_values = list(range(10, 110, 10))

max_errors = []
significant_counts = []    # NEW

print("Power method with deflation over multiple k values:")

for n_eigs in k_values:
    print(f"Computing k = {n_eigs}")

    times = []
    collected_eigs = None

    for _ in range(n):
        B = L_sym.copy()
        eigs = []
        np.random.seed(40)

        start = time.time()

        for j in range(n_eigs):
            q0 = np.random.randn(B.shape[0])
            lam, v = power_method(B, q0)
            eigs.append(lam)
            v = v / np.linalg.norm(v)
            B = B - lam * np.outer(v, v)

        end = time.time()
        times.append(end - start)
        collected_eigs = eigs

    print(f"  Avg Time = {np.mean(times):.4f}s")

    # relative_error now returns (errors, count)
    err, count = relative_error(r1[:n_eigs], collected_eigs, False)

    max_errors.append(max(err))
    significant_counts.append(count)      # store count of errors > 1e-3

# Plot 1: Max error
plt.figure(figsize=(8, 5))
plt.plot(k_values, max_errors, marker="o")
plt.xlabel("Number of eigenvalues computed (k)")
plt.ylabel("Max Relative Error")
plt.title("Max Relative Error vs Number of Eigenvalues (Power Method + Deflation)")
plt.grid(True)
plt.show()

# Plot 2: Significative errors
plt.figure(figsize=(8, 5))
plt.plot(k_values, significant_counts, marker="o")
plt.xlabel("Number of eigenvalues computed (k)")
plt.ylabel("Count of Relative Errors > 1e-3")
plt.title("Significant Errors Count vs Number of Eigenvalues")
plt.grid(True)
plt.show()