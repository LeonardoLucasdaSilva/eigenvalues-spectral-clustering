import numpy as np

def relative_error(v1,v2):
    v1 = sorted(v1, reverse=True)
    v2 = sorted(v2, reverse=True)
    error = []
    for i in range(len(v1)):
        if v1[i]<1e-14:
            if v2[i]<1e-14:
                error.append(0)
            else:
                error.append(np.linalg.norm(v1[i] - v2[i])/np.linalg.norm(v2[i]))
        else:
            error.append(np.linalg.norm(v1[i] - v2[i]) / np.linalg.norm(v2[i]))

    print("Significative errors:")
    print([(i, x) for i, x in enumerate(error) if x > 0.001])