# 🌸 Iris Flower Classification & AI Data Analytics Pipeline

A comprehensive, production-grade Machine Learning classification application built with **Python**, **Scikit-Learn**, and **Streamlit**. This project features an end-to-end Machine Learning pipeline supporting both the classic **Iris Benchmark Dataset** (150 samples) and a multi-class **E-Commerce Dataset** (1200 samples).

---

## 🌟 Key Features

- **Dual Dataset ML Pipeline:** Supports Iris Benchmark dataset (Setosa, Versicolor, Virginica) & Custom E-Commerce Dataset (Order Status classification).
- **Standard Feature Scaling:** Uses `StandardScaler` ($Mean = 0, Variance = 1$) to ensure unbiased distance metrics.
- **Structural Integrity:** Implements 80/20 train-test splitting with mandatory data shuffling to eliminate sequential order bias.
- **KNN Hyperparameter Tuning (Elbow Method):** Evaluates $K = 1 \dots 100$ error rates, identifying the optimal $K$ value to balance overfitting and underfitting.
- **Diagnostic Metrics & Confusion Matrix:** Displays True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN) along with Accuracy, Precision, Recall, and F1-Score.
- **"Accuracy Mirage" Demonstration:** Demonstrates why raw accuracy is deceptive on imbalanced datasets and why F1 Score is the reliable metric.
- **Interactive Streamlit Web Dashboard:** Modern, glassmorphic interactive web app with live custom predictions, dynamic parameter tuning sliders, and visual charts.

---

## 🏗️ Project Architecture (IPO Framework)

```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│     INPUT STAGE      │ ─► │    PROCESS STAGE     │ ─► │    OUTPUT STAGE      │
│                      │    │                      │    │                      │
│  • Dataset Load      │    │  • Shuffle Data      │    │  • Live Predictions  │
│  • Feature Selection │    │  • 80/20 Train-Test  │    │  • Confusion Matrix  │
│  • StandardScaler    │    │  • K=1..100 Tuning   │    │  • Accuracy Mirage   │
│    (Mean=0, Var=1)   │    │  • KNN fit/predict   │    │  • Precision/Recall  │
└──────────────────────┘    └──────────────────────┘    │  • F1 Score Metric   │
                                                         └──────────────────────┘
```

---

## 📁 Repository Structure

```text
├── app.py                      # Interactive Streamlit Web Application
├── train_iris_knn.py           # Core Machine Learning Pipeline Script
├── iris_dataset.csv            # Iris Benchmark Dataset (CSV)
├── iris_dataset.xlsx           # Iris Benchmark Dataset (Excel)
├── Dataset for Data Analytics.xlsx # E-Commerce Analytics Dataset
├── elbow_iris.png              # Generated Elbow Curve (Iris)
├── cm_iris.png                 # Generated Confusion Matrix (Iris)
├── elbow_custom.png            # Generated Elbow Curve (E-Commerce)
├── cm_custom.png               # Generated Confusion Matrix (E-Commerce)
├── .gitignore                  # Git Ignore Configuration
└── README.md                   # Project Documentation
```

---

## ⚙️ Installation & Usage

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Install Required Packages
```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit plotly openpyxl
```

### 3. Run the ML Pipeline (CLI)
To run the automated machine learning pipeline and generate evaluation plots:
```bash
python train_iris_knn.py
```

### 4. Launch the Web Application
To start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📈 Model Performance Metrics

### Iris Benchmark Dataset
- **Accuracy:** 96.67%
- **Precision:** 96.97%
- **Recall:** 96.67%
- **F1-Score:** 96.66%
- **Optimal K:** K = 1

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
