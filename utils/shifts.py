import numpy as np

def wilkinson_shift(a_mm, a_m1m1, b_m1):

    d = (a_m1m1 - a_mm) / 2
    sign = 1 if d >= 0 else -1
    mu = a_mm - b_m1**2 / (abs(d) + np.sqrt(d*d + b_m1*b_m1)) * sign
    return mu