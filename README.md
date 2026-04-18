# 🧠 Pattern Recognition & Statistical Learning Suite

This repository features a comprehensive suite of algorithms for **Pattern Recognition, Machine Learning, and Computer Vision**, implemented in Python. The project bridges the gap between raw data processing and advanced statistical decision theory.

---

## 🚀 Module Overview

### 1. 🖐️ Computer Vision: Rock-Paper-Scissors Classifier
An end-to-end vision system designed to classify hand gestures based on geometric feature extraction.
* **Image Processing Pipeline:** HSV color space segmentation, binary masking, median filtering, and automated bounding-box cropping.
* **Feature Engineering:** Manual extraction of key descriptors:
    * **Perimeter:** The arc length of the hand contour.
    * **Solidity:** The ratio of the contour area to its convex hull area.
* **Models:** Comparison between a **Bayesian Classifier** for 3-class recognition and a **Linear Least Squares (Desired Output)** method for binary classification.

### 2. 📊 Statistical Decision Theory (GMM Modeling)
Advanced analysis using synthetic data generated from **Gaussian Mixture Models (GMM)** to simulate complex class overlaps.
* **Bayesian Decision Theory:** Minimizing total risk and error rates by integrating a cost matrix ($C_{12}, C_{21}$) to penalize specific misclassifications.
* **Neyman-Pearson Criterion:** Designing optimal classifiers with fixed constraints on Type II error ($\beta$) while minimizing Type I error ($\alpha$).
* **Wald’s Sequential Probability Ratio Test (SPRT):** A dynamic decision-making model where the number of samples is not fixed, but determined by crossing cumulative log-likelihood thresholds.

### 3. 🌀 Clustering Algorithms (Unsupervised Learning)
Exploring data grouping without prior labels using iterative optimization:
* **K-Means (C-Means):** Investigating the impact of initialization (Random vs. Block) on convergence speed and local optima.
* **Expectation-Maximization (EM):** Iterative parameter estimation for GMM distributions.
* **Nonlinear Iterative Decomposition:** Solving non-linearly separable problems (e.g., "square-within-a-circle" distributions) using iterative quadratic boundaries.

---

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **Computer Vision:** `OpenCV (cv2)` for image segmentation and contour analysis.
* **Mathematics:** `NumPy` & `SciPy` for matrix algebra, numerical integration, and statistical distributions.
* **Visualization:** `Matplotlib` for high-fidelity 2D/3D decision space plots and histograms.
* **Evaluation:** `Scikit-learn` for Confusion Matrices and Accuracy metrics.

---

## 📉 Key Insights & Visualizations
The project generates several analytical plots to validate the mathematical models:
* **Decision Regions:** Mapping the 2D feature space to show where the classifier switches decisions.
* **Feature Histograms:** Analyzing the discriminative power of Perimeter vs. Solidity.
* **Wald's Trajectories:** Tracking the "random walk" of log-likelihood ratios until a boundary is reached.
* **Confusion Matrices:** Providing a detailed breakdown of classification precision and recall.

---

## 📂 Project Structure
```text
├── Rock-Paper-Scissors/       # CV classification scripts and datasets
├── Statistical-Decisions/     # Bayesian, Neyman-Pearson, and Wald tests
├── Clustering-Analysis/       # K-Means, EM, and Nonlinear decomposition
└── README.md
