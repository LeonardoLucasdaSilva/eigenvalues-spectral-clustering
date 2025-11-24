import numpy as np
from algorithms.power_method import power_method, shifted_inverse_power_method
from algorithms.jacobi_ciclic import jacobi_cyclic
from algorithms.retrieve_data import matrix_L_sym
from utils.metrics import relative_error
import time
from joblib import Parallel, delayed

# Iris tests

path_iris = "../datasets/iris.csv"
cols_iris = ["sepal_length","sepal_width","petal_length","petal_width"]  # choose which columns matter
filename_iris = "../algorithms/iris.npy"

L_iris = matrix_L_sym(True, 0.01, 'iris')
B = L_iris.copy()


print("Eigenpairs by Python Library:")
start = time.time()
r1 = np.linalg.eigvals(B)
end = time.time()
print("Elapsed time:", end - start, "seconds")
#
# print("QR iteration:")
#
# B = L_iris.copy()
# print(B)
#
# start = time.time()
#
# H,Q = qr_iteration_householder_deflation(B, tol = 1e-8)
#
# end = time.time()
# print("Elapsed time:", end - start, "seconds")
#
# r2 = np.diag(H)
# print(r2)
#
# relative_error(r1,r2)
#
# B = L_iris.copy()
#
k = 150
print(f"Power method with deflation(k = {k}):")

B = L_iris.copy()

start = time.time()
eigs = []
np.random.seed(34)

for j in range(k):
    q0 = np.random.randn(B.shape[0])
    lam, v = power_method(B, q0)
    eigs.append(lam)
    v = v / np.linalg.norm(v)
    B = B - lam * np.outer(v, v)

end = time.time()

print("Elapsed time:", end - start, "seconds")

relative_error(r1[:k],eigs)

print("Jacobi cyclic:")

# B = L_iris.copy()
#
# A,V = jacobi_cyclic(B)
#
# r4 = np.diag(A)
#
# relative_error(r1,r4)
#
# print("Lanczos:")
#
# B = L_iris.copy()
#
# v1 = np.random.rand(B.shape[0])
#
# r5, eigenvectors = lanczos(B,v1, B.shape[0], B.shape[0])
#
# print(r1)
# print(r5)
#
# relative_error(r1,r5, False)