import numpy as np

def power_method(A, q0, maxiter = 1000, tol = 1e-12):
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
        if (np.fabs(nu[1]-nu[0])<tol):

            return nu[1],q

        # Updates the last iteration value
        nu[0] = nu[1]
