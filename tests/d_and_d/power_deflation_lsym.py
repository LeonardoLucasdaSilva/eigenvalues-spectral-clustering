import numpy as np
import time
import matplotlib.pyplot as plt

from algorithms.power_method import power_method
from algorithms.retrieve_data import matrix_L_sym
from utils.metrics import relative_error, orthogonality_measure

# Build matrix
L_sym = matrix_L_sym(True, 0.01, 'd_and_d')

# Ground truth eigenvalues
r1,v1 = np.linalg.eig(L_sym)

r_sorted = np.sort(r1)[::-1]   # descending if you prefer

# diffs already computed
diffs = np.abs(np.diff(r_sorted[:120]))

k = 10
closest_indices = np.argsort(diffs)[:k]

print("Closest eigenvalue pairs:")
for idx in closest_indices:
    print(
        f"i = {idx:3d} | "
        f"λ[i] = {r_sorted[idx]: .15f} | "
        f"λ[i+1] = {r_sorted[idx+1]: .15f} | "
        f"diff = {diffs[idx]:.3e}"
    )

# Number of samples per run
n = 1

# Range of k values
k_values = list(range(10, 110, 10))

max_errors = []
significant_counts = []

print("Power method with deflation over multiple k values:")

for n_eigs in k_values:
    print(f"Computing k = {n_eigs}")

    times = []

    for _ in range(n):
        B = L_sym.copy()
        eigenvalues = []
        eigenvectors = []
        np.random.seed(40)

        start = time.time()

        for j in range(n_eigs):
            q0 = np.random.randn(B.shape[0])
            lam, v = power_method(B, q0)
            eigenvalues.append(lam)
            eigenvectors.append(v)
            v = v / np.linalg.norm(v)
            B = B - lam * np.outer(v, v)

        end = time.time()
        times.append(end - start)

    print(f" Average time ({n} samples)= {np.mean(times):.4f}s")

    # relative_error now returns (errors, count)
    err, count = relative_error(r1[:n_eigs], eigenvalues, False)

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