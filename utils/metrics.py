import numpy as np
import matplotlib.pyplot as plt


def relative_error(v1, v2, title, plot=True):
    v1 = sorted(v1, reverse=True)
    v2 = sorted(v2, reverse=True)

    error = []
    for i in range(len(v1)):
        num = v1[i] - v2[i]
        den = v1[i]

        # Se a parte real é ~0 para ambos, erro = 0
        if abs(np.real(num)) < 1e-10 and abs(np.real(den)) < 1e-10:
            error.append(1e-15)
        else:
            error.append(np.linalg.norm(num) / np.linalg.norm(den))

    if plot:
        plt.figure()
        plt.plot(error, marker='o')
        plt.yscale("log")
        plt.title(f"Relative Error per Component ({title})")
        plt.xlabel("Component Index")
        plt.ylabel("Relative Error")
        plt.grid(True)
        plt.show()

    print("Significative errors:")
    significative_errors = [(v1[i],v2[i], i, x) for i, x in enumerate(error) if x > 1e-3]
    print(significative_errors)

    return error, len(significative_errors)


def orthogonality_measure(u, v):
    u = np.asarray(u)
    v = np.asarray(v)

    dot = np.dot(u, v)
    norms = np.linalg.norm(u) * np.linalg.norm(v)

    cos_theta = dot / norms
    print(cos_theta)
    return abs(cos_theta)