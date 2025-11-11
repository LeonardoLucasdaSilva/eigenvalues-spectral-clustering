import numpy as np

def potencia(A, q0, maxiter = 1000, tol = 1e-12):

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

      print("Convergiu")
      print(nu[1])

      break;

    # Updates the last iteration value
    nu[0] = nu[1]


n = 4

A = np.array([
    [10.0, 3.0, 0.0, 0.0],
    [3.0, 7.0, 2.0, 0.0],
    [0.0, 2.0, 4.0, 1.0],
    [0.0, 0.0, 1.0, 2.0]
], dtype=float)

print(np.linalg.eigvals(A))

q0 = np.ones(n)

potencia(A,q0)
