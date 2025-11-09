"""
givens.py

Provides givens transformation to higher-level algorithms
Most of these algorithms are based on the pseudocodes shown in Golub, 2023 - Matrix Computations

Functions:
    givens_cs(x): Computes the parameters c and s for a Givens rotation the direct way
    givens_rho(c,s): Computes the parameter rho that saves the information about the givens rotation through the
    trigonometrical relation between c and s -> c^2 + s^2 = 1
    retrieve_cs(rho): Retrieves the parameters c and s from the parameter rho
    givens_args(xi,xk): Computes the parameters c and s for a Givens rotation the optimized way
"""

import numpy as np
from scipy.linalg.lapack import dgehrd
np.set_printoptions(precision=3, suppress=True, linewidth=150)

def givens_cs(xi,xk):
    """
    Computes the parameters c and s for a Givens rotation the direct way

    Args:
        xi (float): parameter xi
        xk (float): parameter xk

    Returns:
        c (float): cos(theta)
        s (float): sin(theta)
    """

    c = xi/(np.sqrt(xi**2 + xk**2))
    s = -xk/(np.sqrt(xi**2 + xk**2))

    return c,s

def givens_rho(c,s):
    """
    Computes the parameter rho that saves the information about the givens rotation through the trigonometrical
    relation between c and s -> c^2 + s^2 = 1

    Args:
        c (float): cos(theta)
        s (float): sin(theta)

    Returns:
        rho (float): parameter rho
    """

    if c == 0:
        rho = 1
    elif np.abs(s) < np.abs(c):
        rho = np.sign(s)*(s/2)
    else:
        rho = np.sign(s)*(c/2)

    return rho

def retrieve_cs(rho):
    """
    Retrieves the parameters c and s from the parameter rho

    Args:
        rho (float): parameter rho

    Returns:
        c (float): cos(theta)
        s (float): sin(theta)
    """

    if rho == 1:
        c = 0
        s = 1
    elif np.abs(rho) < 1:
        s = 2*rho
        c = np.sqrt(1-s**2)
    else:
        c = 2/rho
        s = np.sqrt(1-c**2)

    return c,s


def givens_args(xi,xk):
    """
    Computes the parameters c and s for a Givens rotation the optimized way

    Args:
        xi (float): parameter xi
        xk (float): parameter xk

    Returns:
        c (np.array): cos(theta)
        s (np.array): sin(theta)
    """

    if xk == 0:
        c = 1
        s = 0

    else:
        if np.abs(xk) > np.abs(xi):
            tal = -xi/xk
            s = 1/np.sqrt(1 + tal**2)
            c = s*tal
        else:
            tal = -xk/xi
            c = 1/np.sqrt(1 + tal**2)
            s = c*tal

    return c,s