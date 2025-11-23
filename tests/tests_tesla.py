import numpy as np
from algorithms.arnoldi import arnoldi
from algorithms.lanczos import lanczos
from algorithms.power_method import power_method, shifted_inverse_power_method
from algorithms.qr_iteration import qr_iteration_householder, qr_iteration_householder_deflation
from algorithms.jacobi_ciclic import jacobi_cyclic
from algorithms.retrieve_data import matrix_L
from utils.metrics import relative_error
import os
import time
from joblib import Parallel, delayed

np.set_printoptions(precision=4, suppress=True, linewidth=450,threshold=10)

# Tesla deliveries tests

path_tesla = "../datasets/tesla_deliveries_dataset_2015_2025.csv"
cols_tesla = ["Estimated_Deliveries","Production_Units","Avg_Price_USD","Battery_Capacity_kWh","Range_km","CO2_Saved_tons","Charging_Stations"] # choose which columns matter
filename_tesla = "../algorithms/tesla.npy"
L_tesla = matrix_L(path_tesla,cols_tesla,filename_tesla,True, 0.01)

B = L_tesla.copy()
print("Eigenvalues by Python Library:")
r1 = np.linalg.eigvals(B)
print(r1)

# for k in range(2,20):
#
#     print(f"Power method with QR(k = {k}):")
#
#     B = L_tesla.copy()
#
#     start = time.time()
#
#     H, Q = qr_iteration_householder(B, max_iter=1,tol=5e-3)
#     diag = np.diag(H)
#     print(diag)
#
#     results = Parallel(n_jobs=-5)(
#         delayed(shifted_inverse_power_method)(B, Q[i,:], diag[i])
#         for i in range(k)
#     )
#
#     end = time.time()
#
#     r3 = [float(val) for val, vec in results]
#
#     print("Elapsed time:", end - start, "seconds")
#
#     print(r1[:k])
#     print(r3)
#
#     relative_error(r1[:k],r3)

k = 100

print(f"Power method with deflation(k = {k}):")

B = L_tesla.copy()

start = time.time()
eigs = []

for j in range(k):
    q0 = np.random.randn(B.shape[0])
    lam, v = power_method(B, q0)
    eigs.append(lam)
    v = v / np.linalg.norm(v)
    B = B - lam * np.outer(v, v)

end = time.time()

print("Elapsed time:", end - start, "seconds")

print(r1[:k])
print(eigs)

relative_error(r1[:k],eigs)
