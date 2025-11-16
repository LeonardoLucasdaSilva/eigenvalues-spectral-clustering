"""
givens.py

Provides Jacobi transformation to higher-level algorithms

Functions:
    jacobi_cs(xi,xk): Computes the parameters c and s for a Jacobi rotation the optimized way
"""

import numpy as np
from scipy.linalg.lapack import dgehrd
np.set_printoptions(precision=3, suppress=True, linewidth=150)

def jacobi_args(app,apq,aqq):
    """
    Computes the parameters c and s for a Jacobi rotation

    Args:
        aqq (float): element a_qq of the matrix
        app (float): element a_pq of the matrix
        apq (float): element a_pp of the matrix

    Returns:
        c (float): cos(theta)
        s (float): sin(theta)
    """

    if apq == 0:
        return 1.0,0.0

    tau = (aqq-app)/(2*apq)
    tplus = -tau+np.sqrt(1 + tau**2)
    tminus = -tau - np.sqrt(1 + tau ** 2)

    if np.abs(tplus) > np.abs(tminus):
        c = 1/np.sqrt(1 + tminus**2)
        s = c*tminus
    else:
        c = 1/np.sqrt(1 + tplus**2)
        s = c*tplus

    return c,s




