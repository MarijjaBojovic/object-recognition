import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det

N = 500  # broj uzoraka po klasi

# Klasa 1: unutar kvadrata
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Generisanje uniformnih tačaka
u = np.random.rand(N)
v = np.random.rand(N)

# Skaliranje da padne u kvadrat
X = np.zeros((N, 2))
X[:,0] = x_min + u * (x_max - x_min)  # X-koordinata
X[:,1] = y_min + v * (y_max - y_min)

# Klasa 2: spoljašnji krug
r1 = 3
r2 = 4
theta2 = 2 * np.pi * np.random.rand(N)
radii2 = np.sqrt(np.random.rand(N) * (r2**2 - r1**2) + r1**2)
Y = np.column_stack((radii2 * np.cos(theta2), radii2 * np.sin(theta2)))

plt.scatter(X[:,0], X[:,1], color='red', label='Klasa 1 (kvadrat)')
plt.scatter(Y[:,0], Y[:,1], color='blue', label='Klasa 2 (krug)')
plt.title('Nelinearno separabilne klase - kvadrat i kružni raspored')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# data = np.vstack((X, Y))
# ra = 0.5
# rb = 2 * ra
# alpha = 1 / (2 * ra**2)
# beta = 1 / (2 * rb**2)
# epsilon = 0.05
# grid_size = 50
# xg = np.linspace(np.min(data[:,0])-1, np.max(data[:,0])+1, grid_size)
# yg = np.linspace(np.min(data[:,1])-1, np.max(data[:,1])+1, grid_size)
# Xg, Yg = np.meshgrid(xg, yg)
# grid = np.column_stack((Xg.ravel(), Yg.ravel()))
# M = np.zeros(grid.shape[0])
# for i, g in enumerate(grid):
#     diff = data - g
#     M[i] = np.sum(np.exp(-np.sum(diff**2, axis=1) * alpha))
#
# M_original_max = np.max(M)
# centers = []
# while True:
#     idx = np.argmax(M)
#     M_current_max = M[idx]
#     if M_current_max < epsilon * M_original_max:
#         break
#     new_center = grid[idx]
#     centers.append(new_center)
#     # Smanjujemo potencijal u okolini novog centra
#     for i, gp in enumerate(grid):
#         dist_sq = np.sum((gp - new_center)**2)
#         M[i] -= M_current_max * np.exp(-dist_sq * beta)
#
# centers = np.array(centers)
#
# plt.figure()
# plt.scatter(data[:,0], data[:,1], alpha=0.3, label='Podaci')
# plt.scatter(centers[:,0], centers[:,1], label='Subtractive Clustering Centri')
# plt.title(f'Subtractive Clustering (broj klastera = {len(centers)})')
# plt.legend()
# plt.grid()
# plt.show()


X = X.T
Y = Y.T

pom = np.random.rand(2 * N)
# Dajemo po 100 odbriraka koji zapravo pripadaju klasi
X1 = X[:,:100]
Y1 = Y[:,:100]

# Za ostale odbirke generišemo nasumične vrednosti pom, kao prilikom primene C-mean klasterizacije
for i in range(100, N):
    if pom[i] < 0.5:
        X1 = np.hstack((X1, X[:, [i]]))
    else:
        Y1 = np.hstack((Y1, X[:, [i]]))

for i in range(N, 2 * N):
    if pom[i] < 0.5:
        X1 = np.hstack((X1, Y[:, [i - N]]))
    else:
        Y1 = np.hstack((Y1, Y[:, [i - N]]))

plt.figure()
plt.plot(X1[0], X1[1], 'rx', label='Klaster 1')
plt.plot(Y1[0], Y1[1], 'bx', label='Klaster 2')
plt.axis('equal')
plt.legend()
plt.title('Inicijalna klasterizacija')
plt.show()

lmax = 100
l = 0
reclassified = True
#Iterativna kvadratna dekompozicija
while l < lmax and reclassified:
    M1 = np.mean(X1, axis=1, keepdims=True)
    M2 = np.mean(Y1, axis=1, keepdims=True)
    S1 = np.cov(X1)
    S2 = np.cov(Y1)
    P1 = X1.shape[1] / (2 * N)
    P2 = Y1.shape[1] / (2 * N)

    X1pom, Y1pom = [], []
    reclassified = False
    # Reklasifikacija odbiraka
    for i in range(np.shape(X1)[1]):
        x = X1[:, i].reshape(2, 1)
        d1 = 0.5 * (x - M1).T @ inv(S1) @ (x - M1) + 0.5 * np.log(det(S1)) - 0.5 * np.log(P1)
        d2 = 0.5 * (x - M2).T @ inv(S2) @ (x - M2) + 0.5 * np.log(det(S2)) - 0.5 * np.log(P2)
        if d1 < d2:
            X1pom.append(X1[:, i])
        else:
            Y1pom.append(X1[:, i])
            reclassified = True

    for i in range(np.shape(Y1)[1]):
        x = Y1[:, i].reshape(2, 1)
        d1 = 0.5 * (x - M1).T @ inv(S1) @ (x - M1) + 0.5 * np.log(det(S1)) - 0.5 * np.log(P1)
        d2 = 0.5 * (x - M2).T @ inv(S2) @ (x - M2) + 0.5 * np.log(det(S2)) - 0.5 * np.log(P2)
        if d1 < d2:
            X1pom.append(Y1[:, i])
            reclassified = True
        else:
            Y1pom.append(Y1[:, i])

    X1 = np.array(X1pom).T
    Y1 = np.array(Y1pom).T

    plt.figure(figsize=(6, 6))
    plt.plot(X1[0], X1[1], 'ro')
    plt.plot(Y1[0], Y1[1], 'bx')
    plt.title(f'Iteracija {l + 1}')
    plt.show()

    l += 1

# Nasumična inicijalizacija
mask_X = np.random.rand(N) < 0.5
X1 = X[:, mask_X]
Y1 = X[:, ~mask_X]

mask_Y = np.random.rand(N) < 0.5
X1 = np.hstack((X1, Y[:, mask_Y]))
Y1 = np.hstack((Y1, Y[:, ~mask_Y]))

plt.figure()
plt.scatter(X1[0], X1[1], color='red', label='Klaster 1')
plt.scatter(Y1[0], Y1[1], color='blue', label='Klaster 2')
plt.axis('equal')
plt.legend()
plt.title('Inicijalna nasumična klasterizacija')
plt.show()

lmax = 100
l = 0
reclassified = True
while l < lmax and reclassified:
    M1 = np.mean(X1, axis=1, keepdims=True)
    M2 = np.mean(Y1, axis=1, keepdims=True)
    S1 = np.cov(X1)
    S2 = np.cov(Y1)
    P1 = X1.shape[1] / (2 * N)
    P2 = Y1.shape[1] / (2 * N)

    X1pom, Y1pom = [], []
    reclassified = False

    for i in range(np.shape(X1)[1]):
        x = X1[:, i].reshape(2, 1)
        d1 = 0.5 * (x - M1).T @ inv(S1) @ (x - M1) + 0.5 * np.log(det(S1)) - 0.5 * np.log(P1)
        d2 = 0.5 * (x - M2).T @ inv(S2) @ (x - M2) + 0.5 * np.log(det(S2)) - 0.5 * np.log(P2)
        if d1 < d2:
            X1pom.append(X1[:, i])
        else:
            Y1pom.append(X1[:, i])
            reclassified = True

    for i in range(np.shape(Y1)[1]):
        x = Y1[:, i].reshape(2, 1)
        d1 = 0.5 * (x - M1).T @ inv(S1) @ (x - M1) + 0.5 * np.log(det(S1)) - 0.5 * np.log(P1)
        d2 = 0.5 * (x - M2).T @ inv(S2) @ (x - M2) + 0.5 * np.log(det(S2)) - 0.5 * np.log(P2)
        if d1 < d2:
            X1pom.append(Y1[:, i])
            reclassified = True
        else:
            Y1pom.append(Y1[:, i])

    X1 = np.array(X1pom).T
    Y1 = np.array(Y1pom).T

    plt.figure(figsize=(6, 6))
    plt.plot(X1[0], X1[1], 'ro')
    plt.plot(Y1[0], Y1[1], 'bx')
    plt.title(f'Iteracija {l + 1}')
    plt.show()

    l += 1