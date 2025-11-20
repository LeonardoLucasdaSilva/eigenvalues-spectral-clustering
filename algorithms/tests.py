import numpy as np
from algorithms.arnoldi import arnoldi
from algorithms.lanczos import lanczos
from algorithms.power_method import power_method, shifted_inverse_power_method
from algorithms.qr_iteration import qr_iteration_householder
from algorithms.jacobi_ciclic import jacobi_cyclic
from algorithms.retrieve_data import matrix_L
from utils.metrics import relative_error

np.set_printoptions(precision=4, suppress=True, linewidth=150)

# Iris tests

path_iris = "../datasets/iris.csv"
cols_iris = ["sepal_length","sepal_width","petal_length","petal_width"]  # choose which columns matter
L_iris = matrix_L(path_iris,cols_iris)
B = L_iris.copy()

print("Eigenpairs by Python Library:")
r1 = np.linalg.eigvals(B)
print(r1)

print("QR iteration:")

B = L_iris.copy()

H,Q = qr_iteration_householder(B)
r2 = np.diag(H)
print(r2)

relative_error(r1,r2)

B = L_iris.copy()

#print(measure(r1,r2))

print("Power method with QR:")

H, Q = qr_iteration_householder(B, tol=5e-3)
diag = np.diag(H)
eigs = []
q0 = np.ones(len(diag))

for d in diag:
    value, vector = shifted_inverse_power_method(B, q0, d)
    eigs.append(value)

eigs = np.array(eigs)

relative_error(r1,eigs)

print(eigs)
