import numpy as np
from scipy.linalg import lu_factor, lu_solve

def power_method(A, q0, maxiter = 1000, tol = 1e-14):
    """
    Find the dominant eigenvalue of a matrix by the power method.

    Args:
    A (n x n matrix): Matrix input
    q0 (n x n matrix): Initial approximation of the eigenvector
    maxiter (int): Maximum number of iterations
    tol (float): Relative error tolerance

    Returns:
    nu[1]: dominant eigenvalue
    v: corresponding eigenvector

    """

    # Normalizes the initial vector
    q = q0/np.linalg.norm(q0)

    nu = np.zeros(2)

    # First approximation
    nu[0] = np.dot(q, A @ q)

    # Run how many iterations as necessary to converge or until break the max iterations number
    for k in range(maxiter):

        z = A @ q
        q = z/np.linalg.norm(z)
        nu[1] = np.dot(q, A @ q)

        # Compare the last two approximations to see if they are close to each other
        if np.fabs(nu[1] - nu[0]) < tol:

            return nu[1],q

        # Updates the last iteration value
        nu[0] = nu[1]

    return nu[1], q

def shifted_inverse_power_method(A, q0, mu, maxiter=1000, tol=1e-10):
    """
    Find the eigenvalue of A closest to a given shift mu using the shifted inverse power method.

    Args:
        A (n x n matrix): Matrix input
        q0 (n vector): Initial approximation of eigenvector
        mu (float): Shift value
        maxiter (int): Maximum number of iterations
        tol (float): Relative error tolerance

    Returns:
        lambda_closest: eigenvalue closest to mu
        q: corresponding eigenvector
    """

    n = A.shape[0]

    # Precompute shifted matrix
    M = A - mu * np.eye(n)

    # LU factorization
    lu, piv = lu_factor(M)

    q = q0 / np.linalg.norm(q0)
    nu = np.zeros(2)

    # First Rayleigh quotient (true eigenvalue estimate)
    nu[0] = np.dot(q, A @ q)

    for k in range(maxiter):

        # Solve (A - muI) z = q  instead of multiplying
        z = lu_solve((lu, piv), q)

        # Normalize
        q = z / np.linalg.norm(z)

        # Compute Rayleigh quotient
        nu[1] = np.dot(q, A @ q)

        # Check convergence
        if np.abs(nu[1] - nu[0]) < tol:
            return nu[1], q

        nu[0] = nu[1]

    return nu[1], q
