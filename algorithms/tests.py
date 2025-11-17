import numpy as np

from algorithms.arnoldi import arnoldi
from algorithms.power_method import power_method
from algorithms.qr_iteration import qr_iteration_householder
from algorithms.jacobi_ciclic import jacobi_cyclic

A = np.array([
    [19, 22, 19, 28, 34],
    [22, 32, 28, 37, 41],
    [19, 28, 44, 41, 51],
    [28, 37, 41, 52, 56],
    [34, 41, 51, 56, 77]
], dtype=float)

print("Eigenpairs by Python Library:")
print(np.linalg.eig(A))

print("Arnoldi method:")

B = A.copy()
v1 = np.ones(5)
H,Q = arnoldi(B,v1,1000)

print(H)
print(Q.T)

print("QR iteration:")

B = A.copy()

H,Q = qr_iteration_householder(B)
print(H)
print(Q)

print("Jacobi cyclic:")

B = A.copy()

A,V = jacobi_cyclic(B)

print(A)
print(V)