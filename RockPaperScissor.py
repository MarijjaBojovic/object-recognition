import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import random

# Funkcija za crop slike ruke
def crop(imgbin):
    # Median filter da ukloni sitni šum
    imgbin_filtered = scipy.ndimage.median_filter(imgbin, size=3)
    # Pronalazak kontura
    contours, _ = cv2.findContours(imgbin_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Najveća kontura
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Crop po granicama
    cropped = imgbin_filtered[y:y + h, x:x + w]
    return cropped

# Funkcija za ekstrakciju feature-a
def extract_features(hand_mask):
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(hand_contour)
    perimeter = cv2.arcLength(hand_contour, False)
    hull = cv2.convexHull(hand_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    return [area, perimeter, solidity]

# Load i ekstrakcija feature-a
folders = ['rock', 'paper', 'scissors']
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
X = []
y = []

for idx, folder in enumerate(folders):
    folder_path = os.path.join('Rock Paper Scissors', folder)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = cv2.imread(os.path.join(folder_path, filename))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_hand = cv2.bitwise_not(mask_green)
            cropped = crop(mask_hand)
            feat = extract_features(cropped)
            X.append(feat)
            y.append(idx)
X = np.array(X)
y = np.array(y)
print(np.sum(y==1))
print(np.sum(y==0))
print(np.sum(y==2))

# Podela na train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
labels = ['rock','paper','scissors']

# Bayes za 3 klase (linearna Gauss raspodela)
classes = [0, 1, 2]
params = {}

for c in classes:
    Xc = X_train[y_train==c]
    mu = np.mean(Xc, axis=0)
    sigma = np.cov(Xc.T)
    params[c] = (mu, sigma)

def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * (x - mu) @ np.linalg.inv(sigma) @ (x - mu).T) / np.sqrt((2*np.pi)**len(mu) * np.linalg.det(sigma))

y_pred = []
for x in X_test:
    likelihoods = [gaussian_pdf(x, *params[c]) for c in classes]
    pred_class = classes[np.argmax(likelihoods)]
    y_pred.append(pred_class)

cm = confusion_matrix(y_test, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.title("Konfuziona matrica – Bayes klasifikacija")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
print(f"Tačnost: {accuracy*100:.2f}%")
print(f"Greška klasifikacije: {error*100:.2f}%")

# Vizualizacija 9 random slika po klasama
fig, axes = plt.subplots(3, 9)
fig.suptitle("Primeri obradjenih slika po klasama (rock, paper, scissors)")

for row, cls in enumerate([0,1,2]):
    folder_path = os.path.join('Rock Paper Scissors', folders[cls])
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    selected_files = random.sample(all_files, min(9, len(all_files)))

    for col, filename in enumerate(selected_files):
        img = cv2.imread(os.path.join(folder_path, filename))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_hand = cv2.bitwise_not(mask_green)
        cropped = crop(mask_hand)
        axes[row, col].imshow(cropped, cmap='gray')
        axes[row, col].axis('off')

plt.tight_layout()
plt.show()

#Maskiranje po klasama
mask_rock = (y == 0)
mask_scissors = (y == 2)
# Feature indeksi
idx_perimeter = 1
idx_solidity = 2
# Ekstrakcija vrednosti
perimeter_rock = X[mask_rock, idx_perimeter]
perimeter_scissors = X[mask_scissors, idx_perimeter]
solidity_rock = X[mask_rock, idx_solidity]
solidity_scissors = X[mask_scissors, idx_solidity]

# Crtanje histograma
plt.figure()
# Perimeter
plt.subplot(1,2,1)
plt.hist(perimeter_rock, bins=15, alpha=0.6, label='rock', color='blue')
plt.hist(perimeter_scissors, bins=15, alpha=0.6, label='scissors', color='red')
plt.title('Histogram - Perimeter')
plt.xlabel('Perimeter')
plt.ylabel('Broj uzoraka')
plt.legend()
# Solidity
plt.subplot(1,2,2)
plt.hist(solidity_rock, bins=15, alpha=0.6, label='rock', color='blue')
plt.hist(solidity_scissors, bins=15, alpha=0.6, label='scissors', color='red')
plt.title('Histogram - Solidity')
plt.xlabel('Solidity')
plt.ylabel('Broj uzoraka')
plt.legend()
plt.show()

# Funkcija za desired output linearni klasifikator
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

def predict_desired(X, W):
    Xh = np.vstack([np.ones((1, X.shape[1])), X])
    g = W.T @ Xh
    return (g >= 0).astype(int).flatten()

# Biramo samo rock (0) i scissors (2)
mask_rs = (y == 0) | (y == 2)

# Biramo osobine: perimeter (1) i solidity (2)
X_rs = X[mask_rs][:, [1, 2]]
y_rs = y[mask_rs]

# Pretvaranje u binarne labele: rock=0, scissors=1
y_rs_bin = (y_rs == 2)

X_train, X_test, y_train, y_test = train_test_split(
    X_rs, y_rs_bin,
    test_size=0.2,
    random_state=42,
    stratify=y_rs_bin
)

X_rock_tr = X_train[y_train == 0].T
X_scissors_tr = X_train[y_train == 1].T
W = desired_output(X_rock_tr, X_scissors_tr)
y_pred = predict_desired(X_test.T, W)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['rock', 'scissors'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix – Desired Output (test)')
plt.show()

acc = accuracy_score(y_test, y_pred)
print(f"Tačnost (Desired Output): {acc*100:.2f}%")
print(f"Greška: {(1-acc)*100:.2f}%")

plt.figure()
plt.scatter(X_rock_tr[0, :], X_rock_tr[1, :], c='blue', label='rock', alpha=0.6)
plt.scatter(X_scissors_tr[0, :], X_scissors_tr[1, :], c='red', label='scissors', alpha=0.6)
x_vals = np.linspace(X_rs[:,0].min(), X_rs[:,0].max(), 200)
v0, v1, v2 = W.flatten()
y_vals = -(v0 + v1*x_vals) / v2
plt.plot(x_vals, y_vals, 'k', label='Decision boundary')
plt.xlabel('Perimeter')
plt.ylabel('Solidity')
plt.title('Linearni klasifikator – metod željenog izlaza')
plt.legend()
plt.grid(True)
plt.show()
