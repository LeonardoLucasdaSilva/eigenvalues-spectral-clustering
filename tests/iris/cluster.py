import numpy as np

from algorithms.power_method import shifted_inverse_power_method
from algorithms.qr_iteration import qr_iteration_householder_shift_deflation
from algorithms.retrieve_data import matrix_L_sym
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

from utils.metrics import relative_error
from sklearn.cluster import KMeans

# Build the L_rw matrix
L_sym = matrix_L_sym(True, 0.01, 'iris')

def matriz_L(A):
    # Soma das linhas (grau)
    degrees = np.sum(A, axis=1)

    # Evitar divisão por zero
    diag = np.zeros_like(degrees)
    mask = degrees > 0
    diag[mask] = degrees[mask]**(-0.5)

    # Laplaciano normalizado: L = D^(-1/2) * A * D^(-1/2)
    L = A * diag[:, None] * diag[None, :]

    return L

def matriz_Y(L, k):
    # Autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # Índices dos k maiores autovalores
    indices = np.argsort(eigenvalues)[-k:]
    X = eigenvectors[:, indices].real

    # Normalizar linhas
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.zeros_like(X)
    nonzero = norms[:,0] != 0
    Y[nonzero, :] = X[nonzero, :] / norms[nonzero]

    return Y

def NG(L,k):
    Y = matriz_Y(L, k)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(Y)

    return kmeans.labels_

labels = (NG(L_sym, 3))

# True Iris labels: 50 setosa, 50 versicolor, 50 virginica
true_labels = np.repeat([0,1,2], 50)

# Compute confusion matrix
cm = confusion_matrix(true_labels, labels)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="PuOr",
            xticklabels=["Cluster 0", "Cluster 1", "Cluster 2"],
            yticklabels=["Setosa", "Versicolor", "Virginica"])

plt.xlabel("Predicted Cluster")
plt.ylabel("True Class")
plt.title("Confusion Matrix for Iris Clustering")
plt.tight_layout()
plt.show()