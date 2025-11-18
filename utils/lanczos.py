import numpy as np

def sturm_count(a,b,x):
    """
    Computes the sign changes in the sturm sequence given by
    p_0 = 1, p_1 = a_x-1, p_k = (a_k - x)p_{k-1}(x) - b_{k-1}**2 p_{k-2}(x)
    in a symmetrical tridiagonal matrix.

    Args:
    a: main diagonal
    b: lower and upper diagonals
    x: point to be evaluated
    """

    n = len(a)
    previous_p = 1
    current_p = a[0]-x

    count = 0

    # As p_0 equals 1, if p_1 is negative then we increase our counter
    if current_p < 0:
        count += 1

    for k in range(1,n):

        next_p = (a[k]-x)*current_p - (b[k-1]**2) * previous_p

        # Due to the properties of the Sturm sequence, we have that
        # counting the number of negative terms is equivalent to
        # counting the number of sign changes

        if current_p * next_p < 0:     # different signs
            count += 1


        #WATCHOUT UNDER AND OVERFLOW
        previous_p = current_p
        current_p = next_p

    return count

def get_interval(a,b):

    m = len(a)
    L = float('inf')
    R = float('-inf')

    for i in range(m):

        left = abs(b[i - 1]) if i - 1 >= 0 else 0.0
        right = abs(b[i]) if i < m - 1 else 0.0
        radius = left + right

        center = a[i]

        L = min(L, center - radius)
        R = max(R, center + radius)

    return L, R


def largest_eigenvalues(a,b,k,tol = 1e-12):
    """
    Finds the largest eigenvalues in a triangular symmetrical matrix

    Args:
        a: main diagonal
        b: lower and upper diagonals
        k: number of eigenvalues
        tol: tolerance parameter
        
    """

    m = len(a)

    # Gershgorin bounds
    L, R = get_interval(a, b)

    eigenvalues = []

    # Get the eigenvectors, starting from the largest
    for j in range(1, k + 1):

        # rank = number of eigenvalues < lambda
        rank = m - j

        # Bisection
        left = L
        right = eigenvalues[-(j-1)] if eigenvalues else R

        while right - left > tol:
            mid = 0.5 * (left + right)
            count = sturm_count(a, b, mid)

            # If count <= rank, we move left up
            if count <= rank:
                left = mid
            else:
                right = mid

        eigenvalues.append(0.5 * (left + right))

    return eigenvalues


def tridiag_eigenvector(a, b, lam):
    m = len(a)
    y = np.zeros(m)

    y[0] = 1.0
    if m > 1:
        y[1] = ((lam - a[0]) * y[0]) / b[0]

    for k in range(2, m):
        y[k] = ((lam - a[k-1]) * y[k-1] - b[k-2] * y[k-2]) / b[k-1]

    return y / np.linalg.norm(y)