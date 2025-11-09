"""
householder.py

Provides householder transformation to higher-level algorithms
Most of these algorithms are based on the pseudocodes shown in Golub, 2023 - Matrix Computations

Functions:
    householder_args(x): Computes the vector and the beta coefficient used on a householder reflection
    householder_column(x, v, beta): Computes the householder transformation over a column vector.
    householder_row(x, v, beta): Computes the householder transformation over a row vector.
    hessenberg_matrix(A):
"""

import numpy as np
from scipy.linalg import hessenberg
np.set_printoptions(precision=3, suppress=True, linewidth=150)

def householder_args(x):
    """
    Computes the vector and the beta coefficient used on a householder reflection

    Args:
        x (np.array): initial vector

    Returns:
        v (np.array): householder reflection vector
        beta (float): beta coefficient
    """

    n = len(x)
    sigma = x[1:n].T @ x[1:n]

    # The value of v is equal to the value of x besides the first entry
    v = np.ones(n)
    v[1:n] = x[1:n]

    # If we have sigma = 0, then x is already a multiple of the canonical vector e_1
    # Then beta = 0, beta = 2 and beta = -2 all satisfy the properties of the householder transformation
    # The choosing of the values of beta is based on convention and numerical stability
    if sigma == 0:
        beta = 0
    elif sigma == 0 and x[0] < 0:
        beta = -2
    else:
        norm_x = np.sqrt(x[0]**2 + sigma)

        # In this case, we can calculate v[0] the usual way without stability issues
        # Note that because both signs are the same there is no cancellation error
        if x[0] <= 0:
            v[0] = x[0] - norm_x
        # In this case we calculate in an alternative way, to avoid the cancellation error
        else:
            v[0] = -sigma / (x[0]+ norm_x)

        # Finally beta is evaluated based on v(1) and sigma instead of v using algebraic steps
        # Also, beta is rescaled because of the normalization of v that follows
        beta = 2*v[0]**2/(sigma+v[0]**2)

        # We normalize v so that v[0] = 1 so we can handle it easily in the next steps of the Householder reflection.
        v = v/v[0]

    return v, beta

def householder_column(x,v,beta):
    """
    Computes the householder transformation over a column vector.

    Args:
        x (np.array): column vector where the householder reflection is computed
        v (np.array): householder reflection vector
        beta (float): beta coefficient

    Returns:
        hx (np.array): column vector transformed by the householder reflection
    """

    hx = x - beta * v * (v @ x)

    return hx

def householder_row(x,v,beta):
    """
    Computes the householder transformation over a row vector.

    Args:
        x (np.array): row vector where the householder reflection is computed
        v (np.array): householder reflection vector
        beta (float): beta coefficient

    Returns:
        hx (np.array): row vector transformed by the householder reflection
    """

    hx = x - beta * (x @ v) * v

    return hx

def hessenberg_matrix(A):
    """
    Computes the upper hessenberg matrix

    Args:
    A (matrix): matrix A

    Returns:
    A (matrix): upper hessenberg matrix
    """

    n = A.shape[0]

    for i in range(n-1):

        v, beta = householder_args(A[i+1:n,i])
        A[i+1:n,i] = householder_column(A[i+1:n,i], v, beta)
        A[i, i+1:n] = householder_row(A[i,i+1:n], v, beta)

    return A

def compare_with_lapack(A, H_user, tol=1e-10):
    """
    Compare user's Hessenberg matrix with LAPACK's (up to sign differences).
    """
    H_lapack = hessenberg(A)

    print(H_lapack)
    print(H_user)

    print(np.allclose(np.abs(H_user), np.abs(H_lapack)))


A = np.random.rand(10, 10)

H = hessenberg_matrix(A)

compare_with_lapack(A, H)
