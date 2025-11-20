import numpy as np
import pandas as pd

def matrix_L(path,cols):
    # Load CSV
    df = pd.read_csv(path, usecols=cols)

    n = len(df)
    A = np.zeros((n, n))
    sigma = 1

    # Fill matrix
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-(np.linalg.norm(df.iloc[i] - df.iloc[j])) ** 2 / (2 * sigma ** 2))

    # Soma das linhas (grau)
    degrees = np.sum(A, axis=1)

    # Evitar divisão por zero
    diag = np.zeros_like(degrees)
    mask = degrees > 0
    diag[mask] = degrees[mask]**(-0.5)

    # Laplaciano normalizado: L = D^(-1/2) * A * D^(-1/2)
    L = A * diag[:, None] * diag[None, :]

    return L