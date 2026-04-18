import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Generisanje uzoraka ---
N = 500
M1, S1 = np.array([-3, -3]), np.array([[1.6, 0.2], [0.2, 1.5]])
M2, S2 = np.array([4, 4]), np.array([[2.1, 0.1], [0.1, 1.9]])
M3, S3 = np.array([-2, 5]), np.array([[1.3, 0.3], [0.3, 3]])

X1 = np.random.multivariate_normal(M1, S1, N).T  # 2xN
X2 = np.random.multivariate_normal(M2, S2, N).T
X3 = np.random.multivariate_normal(M3, S3, N).T

# Funkcija za desired output
def desired_output(X_pos, X_neg):
    Np = X_pos.shape[1]
    Nn = X_neg.shape[1]
    x0 = np.ones((1, Np + Nn))
    x0[0, :Np] *= -1
    X_all = np.concatenate((X_pos*(-1), X_neg), axis=1)
    U = np.concatenate((x0, X_all), axis=0)
    Gamma = np.ones((Np + Nn, 1))
    # Računanje težina
    W = np.linalg.inv(U @ U.T) @ U @ Gamma
    return W

# Kreiranje one-vs-all klasifikatora
W1 = desired_output(X1, np.concatenate((X2,X3),axis=1))
W2 = desired_output(X2, np.concatenate((X1,X3),axis=1))
W3 = desired_output(X3,np.concatenate((X2,X1),axis=1))
W_all = [W1, W2, W3] # 3x3

plt.figure()
colors = ['red', 'green', 'blue']
labels = ['Klasa 1', 'Klasa 2', 'Klasa 3']
for Xi, c, l in zip([X1, X2, X3], colors, labels):
    plt.scatter(Xi[0,:], Xi[1,:], color=c, label=l)
x_range = np.linspace(-10, 10, 200)
for k, W in enumerate(W_all):
    v0, v1, v2 = W.flatten()
    x2 = -(v0 + v1*x_range)/v2
    plt.plot(x_range, x2, label=f'Granica klase {k+1}')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linearni klasifikator – metod željenog izlaza')
plt.legend()
plt.grid(True)
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.show()

# Funkcija za predikciju
def predikcija_desired(X_all, W_all):
    Xh = np.vstack([np.ones((1, X_all.shape[1])), X_all])
    scores = np.array([W.flatten() @ Xh for W in W_all])
    # argmin po redovima daje klasu sa najmanjom greškom
    return np.argmin(scores, axis=0)

# Priprema podataka
X_total = np.concatenate((X1, X2, X3), axis=1)
y_true = np.array([0]*N + [1]*N + [2]*N)

#  Predikcija
y_pred = predikcija_desired(X_total, W_all)

#  Matrica konfuzije
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Klasa 1', 'Klasa 2', 'Klasa 3'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix – Desired Output')
plt.show()

# Podela svake klase(one vs one)

W1 = desired_output(X1, X2)
W2 = desired_output(X2, X3)
W3 = desired_output(X3,X1)
W_all = [W1, W2, W3] # 3x3

plt.figure()
colors = ['red', 'green', 'blue']
labels = ['Klasa 1', 'Klasa 2', 'Klasa 3']
for Xi, c, l in zip([X1, X2, X3], colors, labels):
    plt.scatter(Xi[0,:], Xi[1,:], color=c, label=l)

x_range = np.linspace(-10, 10, 200)  # crtamo samo do sredine

for k, W in enumerate(W_all):
    v0, v1, v2 = W.flatten()
    x2 = -(v0 + v1*x_range)/v2
    plt.plot(x_range, x2, label=f'Granica klase {k+1} i {k+2}')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Linearni klasifikator – metod željenog izlaza- one vs one')
plt.legend()
plt.grid(True)
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.show()


