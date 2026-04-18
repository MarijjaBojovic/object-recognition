import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Broj uzoraka po klasi
N = 500
# Parametri Gaussovih raspodela
M1, S1 = np.array([-3, -3]), np.array([[1.6, 0.2], [0.2, 1.5]])
M2, S2 = np.array([4, 4]), np.array([[2.1, 0.1], [0.1, 1.9]])
M3, S3 = np.array([-2, 5]), np.array([[1.3, 0.3], [0.3, 3]])
M4, S4 = np.array([5, -3]), np.array([[1, 0.3], [0.3, 0.9]])

# Generisanje podataka
X1 = np.random.multivariate_normal(M1, S1, N)  # 2xN
X2 = np.random.multivariate_normal(M2, S2, N)
X3 = np.random.multivariate_normal(M3, S3, N)
X4 = np.random.multivariate_normal(M4, S4, N)

# Crtanje
plt.figure()
plt.scatter(X1[:,0], X1[:,1], color='red', alpha=0.5, label='Class 1')
plt.scatter(X2[:,0], X2[:,1], color='blue', alpha=0.5, label='Class 2')
plt.scatter(X3[:,0], X3[:,1], color='green', alpha=0.5, label='Class 3')
plt.scatter(X4[:,0], X4[:,1], color='yellow', alpha=0.5, label='Class 4')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Četiri Gaussove klase (2D)')
plt.legend()
plt.grid(True)
plt.show()

data = np.vstack((X1,X2,X3,X4))   # (2000 × 2)
N = data.shape[0]

P = [0.25,0.25,0.25,0.25]                              # P_i
M = np.array([[0,0],[1,0],[0,1],[1,1]])                # M_i
Sigma = np.array([np.eye(2) for _ in range(4)])        # Σ_i

def gaussian(x, M, S):
    d = len(x)
    detS = np.linalg.det(S)
    invS = np.linalg.inv(S)
    norm = 1 / np.sqrt((2*np.pi)**d * detS)
    diff = (x - M)
    return norm * np.exp(-0.5 * diff @ invS @ diff)

max_iter = 50
eps = 1e-5
q_old = np.zeros((N, 4))
num_iter = 1

for l in range(max_iter):
    q = np.zeros((N, 4))
    for k in range(N):
        for i in range(4):
            q[k, i] = P[i] * gaussian(data[k], M[i], Sigma[i])
        q[k] /= np.sum(q[k])

    # Provera konvergencije
    if np.linalg.norm(q - q_old) < eps:
        break
    q_old = q.copy()

    for i in range(4):
        Ni = np.sum(q[:, i])
        P[i] = Ni / N
        M[i] = np.sum(q[:, i, None] * data, axis=0) / Ni
        Sigma[i] = np.zeros((2, 2))
        for k in range(N):
            diff = (data[k] - M[i]).reshape(2, 1)
            Sigma[i] += q[k, i] * (diff @ diff.T)
        Sigma[i] /= Ni
    num_iter += 1
print(f"Algoritam je konvergirao posle {num_iter} iteracija.")
print("A priori verovatnoće P_i:")
print(P)

print("\nSrednje vrednosti M_i:")
for i in range(4):
    print(f"M{i+1} =", M[i])

print("\nKovarijacione matrice Σ_i:")
for i in range(4):
    print(f"Σ{i+1} =\n", Sigma[i])

labels = np.argmax(q, axis=1)   # klaster za svaku tačku
plt.figure()
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label='Podaci')
plt.scatter(M[:, 0], M[:, 1], label='Sredine')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Rezultat ml klasterizacije')
plt.legend()
plt.show()

P = [0.1,0.2,0.5,0.2]
M = np.array([[-5,-5],[0,0],[5,5],[9,9]])
Sigma = np.array([
    [[1, 0], [0, 1]],
    [[2, 0.5], [0.5, 2]],
    [[0.5, 0.5], [0.5, 0.6]],
    [[1, 0.1], [0.1, 1]]
])
num_iter = 0

for l in range(max_iter):
    # E-korak
    q = np.zeros((N, 4))
    for k in range(N):
        for i in range(4):
            q[k, i] = P[i] * gaussian(data[k], M[i], Sigma[i])
        q[k] /= np.sum(q[k])

    # Provera konvergencije
    if np.linalg.norm(q - q_old) < eps:
        break
    q_old = q.copy()

    # M-korak
    for i in range(4):
        Ni = np.sum(q[:, i])
        P[i] = Ni / N
        M[i] = np.sum(q[:, i, None] * data, axis=0) / Ni
        Sigma[i] = np.zeros((2, 2))
        for k in range(N):
            diff = (data[k] - M[i]).reshape(2, 1)
            Sigma[i] += q[k, i] * (diff @ diff.T)
        Sigma[i] /= Ni
    num_iter += 1

print(f"Algoritam je konvergirao posle {num_iter} iteracija.")

print("A priori verovatnoće P_i:")
print(P)

print("\nSredine M_i:")
for i in range(4):
    print(f"M{i+1} =", M[i])

print("\nKovarijacione matrice Σ_i:")
for i in range(4):
    print(f"Σ{i+1} =\n", Sigma[i])

labels = np.argmax(q, axis=1)   # klaster za svaku tačku
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5, label='Podaci')
plt.scatter(M[:, 0], M[:, 1], color='red', marker='x', label='Sredine')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Rezultat ml klasterizacije')
plt.legend()
plt.grid(True)
plt.show()