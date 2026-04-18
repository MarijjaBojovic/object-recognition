import numpy as np
import matplotlib.pyplot as plt

# Broj uzoraka po klasi
N = 500

# Parametri Gaussovih raspodela
M1, S1 = np.array([-3, -3]), np.array([[1.6, 0.2], [0.2, 1.5]])
M2, S2 = np.array([4, 4]), np.array([[2.1, 0.1], [0.1, 1.9]])
M3, S3 = np.array([-2, 5]), np.array([[1.3, 0.3], [0.3, 3]])
M4, S4 = np.array([5, -3]), np.array([[1, 0.3], [0.3, 0.9]])

# Generisanje podataka
X1 = np.random.multivariate_normal(M1, S1, N).T  # 2xN
X2 = np.random.multivariate_normal(M2, S2, N).T
X3 = np.random.multivariate_normal(M3, S3, N).T
X4 = np.random.multivariate_normal(M4, S4, N).T

# Crtanje
plt.figure()
plt.scatter(X1[0,:], X1[1,:], color='red', alpha=0.5, label='Class 1')
plt.scatter(X2[0,:], X2[1,:], color='blue', alpha=0.5, label='Class 2')
plt.scatter(X3[0,:], X3[1,:], color='green', alpha=0.5, label='Class 3')
plt.scatter(X4[0,:], X4[1,:], color='yellow', alpha=0.5, label='Class 4')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Četiri Gaussove klase (2D)')
plt.legend()
plt.grid(True)
plt.show()

data = np.hstack((X1,X2,X3,X4))
#Subtractive clustering
#ra = 0.3
#rb = 2 * ra
#alpha = 1/2/(ra ** 2)
#beta = 1/2/ (rb ** 2)
#epsilon = 0.5
#grid_size = 50
#x = np.linspace(-7, 9, grid_size)
#y = np.linspace(-8, 11, grid_size)
#Xg, Yg = np.meshgrid(x, y)
#grid = np.column_stack((Xg.ravel(), Yg.ravel()))
#M = np.zeros(grid.shape[0])
#for i, g in enumerate(grid):
#    diff = data.T - g
#    M[i] = np.sum(np.exp(-np.sum(diff**2, axis=1))/alpha)
#M_original_max = np.max(M)
#centers = []
#while True:
#     idx = np.argmax(M)
#     M_current_max = M[idx]
#     if M_current_max < epsilon * M_original_max:
#         break
#     new_center = grid[idx]
#     centers.append(new_center)
#     for i, gp in enumerate(grid):
#         dist_sq = np.sum((gp - new_center) ** 2)
#         M[i] -= M_current_max * np.exp(-dist_sq/beta)
#     print(M_current_max / M_original_max)
#
# centers = np.array(centers)
# plt.figure()
# plt.scatter(data.T[:,0], data.T[:,1], alpha=0.3, label='Podaci')
# plt.scatter(centers[:,0], centers[:,1],label='Subtractive clustering')
# plt.title(f'Subtractive clustering(broj klastera = {len(centers)})')
# plt.legend()
# plt.grid()
# plt.show()

labels = np.random.randint(0, 4, size=data.shape[1])

X1 = data[:, labels == 0]
X2 = data[:, labels == 1]
X3 = data[:, labels == 2]
X4 = data[:, labels == 3]

plt.figure(figsize=(6,6))
plt.plot(X1[0,:], X1[1,:], 'rx', label='Klasa 1')
plt.plot(X2[0,:], X2[1,:], 'bx', label='Klasa 2')
plt.plot(X3[0,:], X3[1,:], 'gx', label='Klasa 3')
plt.plot(X4[0,:], X4[1,:], 'kx', label='Klasa 4')
plt.title('Nasumična inicijalna klasterizacija (k = 4)')
plt.legend()
plt.grid()
plt.show()

num_runs = 20
lmax = 100
iterations = []

for run in range(num_runs):
    l = 0
    reclassified = True

    # Nasumična početna raspodela klastera
    labels = np.random.randint(0, 4, size=data.shape[1])
    X1 = data[:, labels == 0]
    X2 = data[:, labels == 1]
    X3 = data[:, labels == 2]
    X4 = data[:, labels == 3]

    clusters = [X1, X2, X3, X4]

    # Petlja klasterizacije
    while l < lmax and reclassified:
        # Srednje vrednosti klastera
        means = [np.mean(c, axis=1, keepdims=True) if c.size > 0 else np.zeros((2,1))
                 for c in clusters]

        new_clusters = [[] for _ in range(4)]
        reclassified = False

        # Reklasifikacija svake tačke
        for i in range(data.shape[1]):
            x = data[:, i].reshape(-1,1)
            dists = [np.sum((x - m)**2) for m in means]
            idx = np.argmin(dists)
            new_clusters[idx].append(x.flatten())

        # Pretvaranje u np.array i provera promena
        for j in range(4):
            new_clusters[j] = np.array(new_clusters[j]).T if new_clusters[j] else np.empty((2,0))
            if not np.array_equal(new_clusters[j], clusters[j]):
                reclassified = True

        clusters = new_clusters
        l += 1

    iterations.append(l)

# Rezultati
avg_iterations = np.mean(iterations)
print("Iteracije:", iterations)
print("Prosečan broj iteracija:", avg_iterations)

# Na kraju, clusters[0] = X1, clusters[1] = X2, ...
X1, X2, X3, X4 = clusters

plt.figure()
plt.plot(X1[0,:], X1[1,:], 'rx', label='Klasa 1')
plt.plot(X2[0,:], X2[1,:], 'bx', label='Klasa 2')
plt.plot(X3[0,:], X3[1,:], 'gx', label='Klasa 3')
plt.plot(X4[0,:], X4[1,:], 'kx', label='Klasa 4')
plt.title(f'rand (0,4) inicijalna klasterizacija (  iteracija = {l})')
plt.legend()
plt.grid()
plt.show()

block_size = 200
labels = np.zeros(2000)
for i in range(2000):
    labels[i] = (i // block_size) % 4

X1 = data[:, labels == 0]
X2 = data[:, labels == 1]
X3 = data[:, labels == 2]
X4 = data[:, labels == 3]

plt.figure()
plt.plot(X1[0,:], X1[1,:], 'rx', label='Klasa 1')
plt.plot(X2[0,:], X2[1,:], 'bx', label='Klasa 2')
plt.plot(X3[0,:], X3[1,:], 'gx', label='Klasa 3')
plt.plot(X4[0,:], X4[1,:], 'kx', label='Klasa 4')
plt.title('Nasumična inicijalna klasterizacija')
plt.legend()
plt.grid()
plt.show()

num_runs = 20          # koliko puta pokrećeš algoritam
lmax = 100             # maksimalno iteracija
iterations = []        # ovde pamtimo broj iteracija

for run in range(num_runs):
    l = 0
    reclassified = True
    # Blokovska početna podela: 500 po klasteru
    block_size = np.random.randint(1, 5) * 100
    labels = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        labels[i] = (i // block_size) % 4

    # Originalni klasteri
    X1_orig = data[:, labels == 0]
    X2_orig = data[:, labels == 1]
    X3_orig = data[:, labels == 2]
    X4_orig = data[:, labels == 3]

    clusters = [X1_orig.copy(), X2_orig.copy(), X3_orig.copy(), X4_orig.copy()]

    all_data = np.hstack(clusters)  # svi podaci zajedno

    while l < lmax and reclassified:
        # Srednje vrednosti klastera
        means = [np.mean(c, axis=1, keepdims=True) if c.size > 0 else np.zeros((2,1))
                 for c in clusters]

        new_clusters = [[] for _ in range(4)]
        reclassified = False

        # Reklasifikacija svake tačke
        for i in range(all_data.shape[1]):
            x = all_data[:, i].reshape(-1,1)
            dists = [np.sum((x - m)**2) for m in means]
            idx = np.argmin(dists)
            new_clusters[idx].append(x.flatten())

        # Pretvaranje u np.array i provera promena
        for j in range(4):
            new_clusters[j] = np.array(new_clusters[j]).T if new_clusters[j] else np.empty((2,0))
            if not np.array_equal(new_clusters[j], clusters[j]):
                reclassified = True

        clusters = new_clusters
        l += 1

    iterations.append(l)

# Rezultati
avg_iterations = np.mean(iterations)
print("Iteracije po run:", iterations)
print("Prosečan broj iteracija:", avg_iterations)

# Na kraju, clusters[0] = X1, clusters[1] = X2, ...
X1, X2, X3, X4 = clusters

# Prikaz konačne klasterizacije
plt.figure(figsize=(6,6))
plt.plot(X1[0,:], X1[1,:], 'rx', label='Klasa 1')
plt.plot(X2[0,:], X2[1,:], 'bx', label='Klasa 2')
plt.plot(X3[0,:], X3[1,:], 'gx', label='Klasa 3')
plt.plot(X4[0,:], X4[1,:], 'kx', label='Klasa 4')
plt.title(f'Blokovska inicijalna klasterizacija (poslednja iteracija = {l})')
plt.legend()
plt.grid()
plt.show()