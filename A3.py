import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

#  Generisanje uzoraka
N = 500
M1, S1 = np.array([-3, -3]), np.array([[1.6, 0.2], [0.2, 1.5]])
M2, S2 = np.array([4, 4]), np.array([[2.1, 0.1], [0.1, 1.9]])
M3, S3 = np.array([-2, 5]), np.array([[1.3, 0.3], [0.3, 3]])
X1 = np.random.multivariate_normal(M1, S1, N)
X2 = np.random.multivariate_normal(M2, S2, N)
X3 = np.random.multivariate_normal(M3, S3, N)
# Vizualizacija
plt.figure()
plt.scatter(X1[:,0], X1[:,1], color='red', label='Klasa 1')
plt.scatter(X2[:,0], X2[:,1], color='blue', label='Klasa 2')
plt.scatter(X3[:,0], X3[:,1], color='green', label='Klasa 3')
plt.xlabel('X1'); plt.ylabel('X2')
plt.title('Tri linearno separabilne klase')
plt.legend()
plt.grid(True)
plt.show()

#   Podela na train/test
X = np.concatenate((X1, X2, X3), axis=0)
y = np.array([0]*N + [1]*N + [2]*N)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Procena parametara M i S
def estimacija(X_train, y_train):
    M = []
    S = []
    klase = np.unique(y_train)
    for i in klase:
        Xi = X_train[y_train == i]
        M.append(np.mean(Xi, axis=0))
        S.append(np.cov(Xi.T))
    return np.array(M), np.array(S)

M_est, S_est = estimacija(X_train, y_train)

# Određivanje V
def V(M, S, s=0.5, i_class=0):
    n_classes = len(M)
    i_rest = [i for i in range(n_classes) if i != i_class]
    M_global = np.mean([M[j] for j in i_rest], axis=0)
    S_global = np.mean([S[j] for j in i_rest], axis=0)
    Vi = np.linalg.inv(s * S[i_class] + (1 - s) * S_global) @ (M[i_class] - M_global)
    return Vi

# Računanje yi
def yi(Xi, Vi):
    return Xi @ Vi.T

# Određivanje v0
def optimalni_v0(y_i, y_rest):
    v0_min = -max(np.max(y_i), np.max(y_rest))
    v0_max = -min(np.min(y_i), np.min(y_rest))
    v0_values = np.linspace(v0_min, v0_max, 1000)
    min_errors = np.inf
    best_v0 = None
    for v0 in v0_values:
        errors_i = np.sum(y_i < -v0)
        errors_rest = np.sum(y_rest > -v0)
        total_errors = errors_i + errors_rest
        if total_errors < min_errors:
            min_errors = total_errors
            best_v0 = v0
    return best_v0, min_errors

# Iterativna procedura za sve klase
s_values = np.linspace(0, 1, 50)
V_all = np.zeros((3, 2))
v0_all = np.zeros(3)
greske_s = []

for s in s_values:
    greske_ukupno = 0
    V_temp = []
    v0_temp = []
    for i in range(3):
        Vi = V(M_est, S_est, s, i_class=i)
        Xi = X_train[y_train == i]
        X_rest = X_train[y_train != i]
        y_i = yi(Xi, Vi)
        y_rest = yi(X_rest, Vi)
        v0_i, greske = optimalni_v0(y_i, y_rest)
        greske_ukupno += greske
        V_temp.append(Vi)
        v0_temp.append(v0_i)
    greske_s.append(greske_ukupno)

# Optimalno s
greske_s = np.array(greske_s)
s_opt = s_values[np.argmin(greske_s)]
print("Optimalno s:", s_opt)
print("Minimalan broj grešaka:", np.min(greske_s))

# Računanje konačnih V i v0 za optimalno s
for i in range(3):
    V_all[i] = V(M_est, S_est, s_opt, i_class=i)
    Xi = X_train[y_train == i]
    X_rest = X_train[y_train != i]
    y_i = yi(Xi, V_all[i])
    y_rest = yi(X_rest, V_all[i])
    v0_all[i], _ = optimalni_v0(y_i, y_rest)

# Predikcija
def predikcija(X, V_all, v0_all):
    n_classes = len(V_all)
    scores = np.zeros((X.shape[0], n_classes))
    for i in range(n_classes):
        scores[:, i] = X @ V_all[i] - v0_all[i]
    return np.argmax(scores, axis=1)

y_pred_train = predikcija(X_train, V_all, v0_all)
y_pred_test = predikcija(X_test, V_all, v0_all)

# Konfuzione matrice
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

disp_train = ConfusionMatrixDisplay(cm_train)
disp_train.plot()
plt.title('Train Confusion Matrix')
plt.show()

disp_test = ConfusionMatrixDisplay(cm_test)
disp_test.plot()
plt.title('Test Confusion Matrix')
plt.show()

# Zavisnost gresaka od s
plt.figure()
plt.plot(s_values, greske_s, marker='o')
plt.xlabel('s')
plt.ylabel('Broj grešaka')
plt.title('greske od s')
plt.grid(True)
plt.show()

# Vizualizacija linearnih granica
plt.figure()
colors = ['red', 'blue', 'green']
cmap = ListedColormap(colors)
for i, Xi in enumerate([X1, X2, X3]):
    plt.scatter(Xi[:, 0], Xi[:, 1], color=colors[i], label=f'Klasa {i + 1}')
xlim = plt.xlim()
ylim = plt.ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 300),np.linspace(ylim[0], ylim[1], 300))
Z = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        point = np.array([xx[i, j], yy[i, j]])
        scores = np.array([point @ V_all[k] - v0_all[k] for k in range(3)])
        Z[i, j] = np.argmax(scores)
plt.contourf(xx, yy, Z, alpha=0.1, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Iterativni linearni klasifikator')
plt.legend()
plt.grid(True)
plt.show()

#one vs one
pairs = [(0, 1), (0, 2), (1, 2)]
granice = []

for (c1, c2) in pairs:
    # Uzimamo samo dve klase
    X_i = np.concatenate((X[y == c1], X[y == c2]), axis=0)
    y_i = np.array([0] * N + [1] * N)  # izlaz 0 i 1

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_i, y_i, test_size=0.2, stratify=y_i, random_state=42)
    # Procena parametara
    M_est, S_est = estimacija(X_train, y_train)
    # Iteracija po s
    s_values = np.linspace(0, 1, 50)
    V_all = np.zeros((2, 2))
    v0_all = np.zeros(2)
    greske_s = []
    for s in s_values:
        greske_ukupno = 0
        for i in range(2):
            Vi = V(M_est, S_est, s, i_class=i)
            Xi = X_train[y_train == i]
            X_rest = X_train[y_train != i]
            y_i = yi(Xi, Vi)
            y_rest = yi(X_rest, Vi)
            v0_i, greske = optimalni_v0(y_i, y_rest)
            greske_ukupno += greske
        greske_s.append(greske_ukupno)
    greske_s = np.array(greske_s)
    s_opt = s_values[np.argmin(greske_s)]
    print("Optimalno s:", s_opt)
    print("Minimalan broj grešaka:", np.min(greske_s))
    # Konačni V i v0
    for i in range(2):
        V_all[i] = V(M_est, S_est, s_opt, i_class=i)
        Xi = X_train[y_train == i]
        X_rest = X_train[y_train != i]
        y_i = yi(Xi, V_all[i])
        y_rest = yi(X_rest, V_all[i])
        v0_all[i], _ = optimalni_v0(y_i, y_rest)

    # Predikcija
    y_pred_train = predikcija(X_train, V_all, v0_all)
    y_pred_test = predikcija(X_test, V_all, v0_all)

    # Konfuzione matrice
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    disp_train = ConfusionMatrixDisplay(cm_train)
    disp_train.plot()
    plt.title(f'Train Confusion Matrix ({c1 + 1} vs {c2 + 1})')
    plt.show()

    disp_test = ConfusionMatrixDisplay(cm_test)
    disp_test.plot()
    plt.title(f'Test Confusion Matrix ({c1 + 1} vs {c2 + 1})')
    plt.show()

    plt.figure()

    # Crtanje svih podataka
    plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Klasa 1')
    plt.scatter(X2[:, 0], X2[:, 1], color='blue', label='Klasa 2')
    plt.scatter(X3[:, 0], X3[:, 1], color='green', label='Klasa 3')

    # Uzimamo granicu između prve klase (0) i druge (1)
    # Razlika diskriminantnih funkcija
    W = V_all[0] - V_all[1]
    b = v0_all[0] - v0_all[1]
    x_vals = np.linspace(plt.xlim()[0], plt.xlim()[1], 300)
    # prava: W1*x + W2*y - b = 0
    y_vals = (b - W[0] * x_vals) / W[1]
    plt.plot(x_vals, y_vals, 'k', linewidth=2)
    # Numeracija prave
    mid_x = np.mean(x_vals)
    mid_y = (b - W[0] * mid_x) / W[1]
    pair_index = pairs.index((c1, c2)) + 1
    plt.text(mid_x, mid_y, f'  ({pair_index})', fontsize=12)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Granica odlučivanja ({c1 + 1} vs {c2 + 1})')
    plt.legend()
    plt.grid(True)
    plt.show()