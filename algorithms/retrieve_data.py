import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def matrix_L_sym(normalize, sigma):

    path = "../../datasets/iris.csv"
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]  # choose which columns matter
    filename = "../../algorithms/iris_sym.npy"

    if os.path.exists(filename):
        print("Loading saved matrix...")
        L = np.load(filename)
    else:
        print("Building matrix...") # whatever function builds your matrix
        # Load CSV
        df = pd.read_csv(path, usecols=cols)

        if normalize:
            col_norms = np.linalg.norm(df, axis=0)
            df = df / col_norms

        n = len(df)
        A = np.zeros((n, n))

        # Fill matrix
        for i in range(n):
            for j in range(n):

                A[i, j] = np.exp(-(np.linalg.norm(df.iloc[i] - df.iloc[j])) ** 2 / (2 * sigma ** 2))
                print(f'i = {i}, j = {j}, A[i, j] = {A[i, j]}')

        # Soma das linhas (grau)
        degrees = np.sum(A, axis=1)

        # Evitar divisão por zero
        diag = np.zeros_like(degrees)
        mask = degrees > 0
        diag[mask] = degrees[mask]**(-0.5)

        # Laplaciano normalizado: L = D^(-1/2) * A * D^(-1/2)
        L = A * diag[:, None] * diag[None, :]
        np.save(filename, L)

    return L

def matrix_L_rw(normalize, sigma):

    path = "../../datasets/iris.csv"
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]  # choose which columns matter
    filename = "../../algorithms/iris_rw.npy"

    if os.path.exists(filename):
        print("Loading saved matrix...")
        L = np.load(filename)

    else:
        print("Building matrix...")

        df = pd.read_csv(path, usecols=cols)

        # Normalize columns if requested
        if normalize:
            col_norms = np.linalg.norm(df, axis=0)
            df = df / col_norms

        n = len(df)
        A = np.zeros((n, n))

        # Build affinity matrix
        for i in range(n):
            for j in range(n):
                A[i, j] = np.exp(-(np.linalg.norm(df.iloc[i] - df.iloc[j])) ** 2 / (2 * sigma ** 2))
                print(f'i = {i}, j = {j}, A[i, j] = {A[i, j]}')

        # Degree vector
        degrees = np.sum(A, axis=1)

        # Avoid division by zero
        diag_inv = np.zeros_like(degrees)
        mask = degrees > 0
        diag_inv[mask] = 1.0 / degrees[mask]

        # Random walk matrix: P = D^{-1} A
        L = diag_inv[:, None] * A

        np.save(filename, L)

    return L