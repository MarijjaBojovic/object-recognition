import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

P11, M11, S11 = 0.6, np.array([0.3, 1.7]).reshape(-1, 1), np.array([[1.1, 0.3], [0.3, 1.2]])
P12, M12, S12 = 0.4, np.array([2.2, 0.6]).reshape(-1, 1), np.array([[0.85, -0.4], [-0.4, 0.7]])
P21, M21, S21 = 0.5, np.array([2.8, 3.3]).reshape(-1, 1), np.array([[1.0, 0.55], [0.55, 1.2]])
P22, M22, S22 = 0.5, np.array([4.5, 1.5]).reshape(-1, 1), np.array([[0.75, 0.35], [0.35, 0.9]])

N = 500
# Generisanje uzoraka
f11 = np.random.multivariate_normal(M11.reshape(-1), S11, int(N*P11))
f12 = np.random.multivariate_normal(M12.reshape(-1), S12, int(N*P12))
uzorci1 = np.concatenate((f11, f12),axis=0)
uzorci1 = np.random.permutation(uzorci1)

f21 = np.random.multivariate_normal(M21.reshape(-1), S21, int(N*P21))
f22 = np.random.multivariate_normal(M22.reshape(-1), S22, int(N*P22))
uzorci2 = np.concatenate((f21, f22),axis=0)
uzorci2 = np.random.permutation(uzorci2)

# a) Odbirci na grafiku
plt.figure()
plt.scatter(uzorci1[:,0], uzorci1[:,1], color='blue', alpha=0.6, label='Klasa 1')
plt.scatter(uzorci2[:,0], uzorci2[:,1], color='red', alpha=0.6, label='Klasa 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Uzorci klasa 1 i 2')
plt.legend()
plt.grid()
plt.show()

# b) Teorijski fgv i histrogrami
x = np.arange(-2, 8, 0.1)
y = np.arange(-2, 8, 0.1)
f1 = np.zeros((len(x), len(y)))
f2 = np.zeros((len(x), len(y)))
h = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        X = np.array([x[i], y[j]])
        # PDF klase 1
        f1[i, j] = (P11 * multivariate_normal.pdf(X, mean=M11.reshape(-1), cov=S11) + P12 * multivariate_normal.pdf(X, mean=M12.reshape(-1), cov=S12))
        # PDF klase 2
        f2[i, j] = (P21 * multivariate_normal.pdf(X, mean=M21.reshape(-1), cov=S21) + P22 * multivariate_normal.pdf(X, mean=M22.reshape(-1), cov=S22))
        # Bayesov test
        h[i, j] = -np.log(f1[i, j] / f2[i, j])
plt.figure()
plt.contour(x, y, f1.T, colors='yellow')
plt.contour(x, y, f2.T, colors='black')
plt.scatter(uzorci1[:,0], uzorci1[:,1], color='blue', alpha=0.3)
plt.scatter(uzorci2[:,0], uzorci2[:,1], color='red', alpha=0.3)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='yellow', lw=2, label='Teorijska PDF – klasa 1'),
    Line2D([0], [0], color='black', lw=2, label='Teorijska PDF – klasa 2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', alpha=0.5, label='Uzorci – klasa 1'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', alpha=0.5, label='Uzorci – klasa 2')
]
plt.legend(handles=legend_elements)
plt.title('Teorijske PDF i generisani uzorci')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid()
plt.show()

x = np.linspace(-2, 7, 100)
y = np.linspace(-2, 6, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
# 3d reprezentacija
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, f1, cmap='Blues', alpha=0.6)
ax.plot_surface(X, Y, f2, cmap='Reds', alpha=0.6)
ax.set_title('f1(x) i f2(x)')
ax.set_xlabel('X2')
ax.set_ylabel('X1')
ax.set_zlabel('Gustina')
plt.show()

# Histogram
plt.figure()
plt.subplot(1,2,1)
plt.hist(uzorci1[:,0], bins=30, alpha=0.5, color='blue', label='Klasa 1')
plt.hist(uzorci2[:,0], bins=30, alpha=0.5, color='red', label='Klasa 2')
plt.title('X1')
plt.legend()
plt.grid()
plt.subplot(1,2,2)
plt.hist(uzorci1[:,1], bins=30, alpha=0.5, color='blue', label='Klasa 1')
plt.hist(uzorci2[:,1], bins=30, alpha=0.5, color='red', label='Klasa 2')
plt.title('X2')
plt.legend()
plt.grid()
plt.show()

# v) Bayes klasifikator minimalne greske
# određivanje verovatnoća
P1 = len(uzorci1[0, :])/(len(uzorci1[0, :]) + len(uzorci2[0, :]))
P2 = 1 - P1
podaci = np.concatenate((uzorci2,uzorci1),axis=1)
podaci = np.random.permutation(podaci)
x = np.arange(np.min(podaci[:, 0]), np.max(podaci[:, 0]), 0.03)
y = np.arange(np.min(podaci[:, 1]), np.max(podaci[:, 1]), 0.03)
f1 = np.zeros((len(x), len(y)))
f2 = np.zeros((len(x), len(y)))
h = np.zeros((len(x), len(y)))
T = np.log(P1 / P2)
X1 = np.empty((0, 2))
X2 = np.empty((0, 2))

def gaussian_pdf(X, M, S):
    detS = np.linalg.det(S)
    invS = np.linalg.inv(S)
    f = 1/(2*np.pi*detS**0.5)*np.exp(-0.5*(X - M).T@invS@(X - M))
    return f[0, 0]

for i in range(0, len(x)):
    for j in range(0, len(y)):
        X = np.array([x[i], y[j]]).reshape(1, 2)
        f1[i, j] = P11*gaussian_pdf(X.T, M11, S11)+P12*gaussian_pdf(X.T,M12,S12)
        f2[i, j] = P21*gaussian_pdf(X.T, M21, S21)+P22*gaussian_pdf(X.T,M22,S22)
        h[i, j] = -np.log(f1[i, j] / f2[i, j])
        if h[i, j] < T:
            X1 = np.append(X1, X, axis=0)
        else:
            X2 = np.append(X2, X, axis=0)

plt.figure()
plt.plot(uzorci1[:, 0], uzorci1[:, 1], 'rx', label = 'klasa1')
plt.plot(uzorci2[:, 0], uzorci2[:, 1], 'bx', label = 'klasa2')
plt.plot(X1[:, 0], X1[:, 1], 'r', alpha = 0.1)
plt.plot(X2[:, 0], X2[:, 1], 'b', alpha = 0.1)
plt.contour(x, y, h.T, levels = [T], colors='black')
plt.title('Bajesov test, prostor odluke')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend()
plt.show()

def Greska(x):
    X = x.reshape(1,2)
    f1 = P11*gaussian_pdf(X.T, M11, S11) + P12*gaussian_pdf(X.T, M12, S12)
    f2 = P21*gaussian_pdf(X.T, M21, S21) + P22*gaussian_pdf(X.T, M22, S22)
    h = -np.log(f1/f2)
    return 1 if h < T else 2

err1 = sum(Greska(x)==2 for x in uzorci1)
err2 = sum(Greska(x)==1 for x in uzorci2)
eps1_emp = err1 / len(uzorci1)
eps2_emp = err2 / len(uzorci2)

# Teorijske greške (numerički)
dx = x[1]-x[0]
dy = y[1]-y[0]
eps1_t = np.sum(f1[h>=T]) * dx * dy
eps2_t = np.sum(f2[h<T]) * dx * dy

print(f"Empirijska greska prve vrste: {eps1_emp:.4f}")
print(f"Empirijska greska druge vrste: {eps2_emp:.4f}")
print(f"Teorijska greska prve vrste: {eps1_t:.4f}")
print(f"Teorijska greska druge vrste: {eps2_t:.4f}")

# g) Klasifikator minimalne cene
C12 = 0.1
C21 = 1
C11 = 0
C22 = 0

def Bajes(X):
    f1 = P11*multivariate_normal.pdf(X, mean=M11.reshape(-1), cov=S11) + P12*multivariate_normal.pdf(X, mean=M12.reshape(-1), cov=S12)
    f2 = P21*multivariate_normal.pdf(X, mean=M21.reshape(-1), cov=S21) + P22*multivariate_normal.pdf(X, mean=M22.reshape(-1), cov=S22)
    T = np.log((P2*C21+P1*C11) / (P1*C12+P2*C22))
    h = np.log(f1/f2)
    y_pred = (h < T)
    return y_pred

svi = np.concatenate((uzorci1, uzorci2), axis=0)
y_tacno = np.zeros(svi.shape[0])
y_tacno[len(uzorci1):] = 1
y_pred = Bajes(svi)

x = np.linspace(np.min(svi[:,0])-1, np.max(svi[:,0])+1, 200)
y = np.linspace(np.min(svi[:,1])-1, np.max(svi[:,1])+1, 200)
X, Y = np.meshgrid(x, y)
XY = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
Z = Bajes(XY).reshape(X.shape)

plt.figure()
plt.scatter(uzorci1[:,0], uzorci1[:,1], color='blue', alpha=0.6, label='Klasa 1')
plt.scatter(uzorci2[:,0], uzorci2[:,1], color='red', alpha=0.6, label='Klasa 2')
plt.contour(X, Y, Z, levels=[0.5], colors='black')  # linija odluke
plt.title('Bajesov klasifikator minimalne cene')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.xlim([-5,10])
plt.ylim([-5,10])
plt.show()

# d) Neyman-Pearson
def NP(X, y, beta_zeljeno=0.1):
    # Računanje verovatnoća
    f1 = P11 * multivariate_normal.pdf(X, mean=M11.reshape(-1), cov=S11) + P12 * multivariate_normal.pdf(X, mean=M12.reshape(-1), cov=S12)
    f2 = P21 * multivariate_normal.pdf(X, mean=M21.reshape(-1), cov=S21) + P22 * multivariate_normal.pdf(X, mean=M22.reshape(-1), cov=S22)
    h = np.log(f1 / f2)
    h_klasa1 = h[y == 0]  # log-ratio vrednosti samo za klasu 1
    h_klasa1_s = np.sort(h_klasa1)  # sortiramo
    index_prag = int(len(h_klasa1_s) * beta_zeljeno)
    T_NP = h_klasa1_s[index_prag]  # prag NP
    y_pred = (h < T_NP)
    alfa_e = np.sum((y_pred == 0) & (y == 1)) / np.sum(y == 1)
    beta_e = np.sum((y_pred == 1) & (y == 0)) / np.sum(y == 0)
    return T_NP, alfa_e, beta_e

beta_vred = np.linspace(0.01, 0.1, 10)
alfa_emp = []
beta_emp = []
T_vred= []

for b in beta_vred:
    T_NP, a_e, b_e = NP(svi, y_tacno, beta_zeljeno=b)
    alfa_emp.append(a_e)
    beta_emp.append(b_e)
    T_vred.append(T_NP)

# Ispis rezultata
for i in range(len(beta_vred)):
    print(f"e2_zadato={beta_vred[i]:.2f}, e1_emp={alfa_emp[i]:.3f}, e2_emp={beta_emp[i]:.3f}")

# Grafički prikaz
plt.figure()
plt.plot(beta_vred, alfa_emp, 'o-', linewidth=2)
plt.xlabel('epsilon2')
plt.ylabel('epsilon1')
plt.title('greske')
plt.grid(True)
plt.show()
# Vizualizacija
plt.figure()
plt.scatter(uzorci1[:,0], uzorci1[:,1], color='blue', alpha=0.6, label='Klasa 1')
plt.scatter(uzorci2[:,0], uzorci2[:,1], color='red', alpha=0.6, label='Klasa 2')

# Crtanje linije NP praga
x_vals = np.linspace(np.min(svi[:,0]), np.max(svi[:,0]), 200)
y_vals = np.linspace(np.min(svi[:,1]), np.max(svi[:,1]), 200)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
XY_grid = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
f1_grid = P11 * multivariate_normal.pdf(XY_grid, mean=M11.reshape(-1), cov=S11) + P12 * multivariate_normal.pdf(XY_grid, mean=M12.reshape(-1), cov=S12)
f2_grid = P21 * multivariate_normal.pdf(XY_grid, mean=M21.reshape(-1), cov=S21) + P22 * multivariate_normal.pdf(XY_grid, mean=M22.reshape(-1), cov=S22)
h_grid = np.log(f1_grid / f2_grid).reshape(X_grid.shape)
plt.contour(X_grid, Y_grid, h_grid, levels=[T_NP], colors='black', linewidths=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Neyman-Pearson klasifikacija')
plt.legend()
plt.xlim([-5,10])
plt.ylim([-5,10])
plt.grid()
plt.show()

# dj) Sekvencijalno-Wald
def Wald(uzorci, alpha, beta, plot=False):
    logA = np.log((1 - beta) / alpha)
    logB = np.log(beta / (1 - alpha))
    cum_logL = 0
    niz = []
    for n, x in enumerate(uzorci, start=1):
        f1 = P11*gaussian_pdf(x, M11, S11) + P12*gaussian_pdf(x, M12, S12)
        f2 = P21*gaussian_pdf(x, M21, S21) + P22*gaussian_pdf(x, M22, S22)
        cum_logL += np.log(f1 / f2)
        niz.append(cum_logL)
        if cum_logL >= logA or cum_logL <= logB:
            break
    if cum_logL >= logA:
        odluka = 1
    elif cum_logL <= logB:
        odluka = 2
    else:
        odluka = 0  # nijedna granica nije predjena

    if plot:
        niz = np.array(niz)
        plt.plot(np.arange(1,len(niz)+1), niz, marker='o', label='Kumulativni log')
        plt.axhline(logA, color='green', linestyle='--', label='Gornja granica logA')
        plt.axhline(logB, color='red', linestyle='--', label='Donja granica logB')
        plt.xlabel('Br uzoraka')
        plt.ylabel('Kumulativni log')
        plt.title(f'Waldов test (α={alpha}, β={beta})')
        plt.legend()
        plt.grid(True)
        plt.show()
    return n, odluka

# presecan br uzoraka za vise eksperimenata
def prosek(uzorci, alpha, beta, Nexp=200):
    broj = []
    for i in range(Nexp):
        seq = np.random.permutation(uzorci)
        n, _ = Wald(seq, alpha, beta)
        broj.append(n)
    return np.mean(broj)

# Zavisnost N od α (fiksno β=0.05)
alfa_vrednosti = np.linspace(0.01, 0.3, 10)
beta_fix = 0.05
N_alfa1 = [prosek(uzorci1, a, beta_fix) for a in alfa_vrednosti]
N_alfa2 = [prosek(uzorci2, a, beta_fix) for a in alfa_vrednosti]

plt.figure()
plt.plot(alfa_vrednosti, N_alfa1, 'o-', label='klasa 1')
plt.plot(alfa_vrednosti, N_alfa2, 's-', label='klasa 2')
plt.xlabel('greska prve vrste α')
plt.ylabel('prosecan broj uzoraka N')
plt.title('Waldов test:  N(α)')
plt.grid(True)
plt.legend()
plt.show()

# Zavisnost N od β (fiksno α=0.05)
beta_vrednosti = np.linspace(0.01, 0.3, 10)
alpha_fix = 0.05
N_beta1 = [prosek(uzorci1, alpha_fix, b) for b in beta_vrednosti]
N_beta2 = [prosek(uzorci2, alpha_fix, b) for b in beta_vrednosti]

plt.figure()
plt.plot(beta_vrednosti, N_beta1, 'o-', label='klasa 1')
plt.plot(beta_vrednosti, N_beta2, 's-', label='klasa 2')
plt.xlabel('Greska druge vrste β')
plt.ylabel('prosecan broj uzoraka N')
plt.title('Waldов test:  N(β)')
plt.grid(True)
plt.legend()
plt.show()

# sekvenca sa grafikom
seq1 = np.random.permutation(uzorci1)
Wald(seq1, alpha=0.05, beta=0.05, plot=True)

seq2 = np.random.permutation(uzorci2)
Wald(seq2, alpha=0.05, beta=0.05, plot=True)
