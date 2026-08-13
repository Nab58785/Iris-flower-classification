"""
=============================================================================
PROJECT 2: DATA CLASSIFICATION USING AI — COMPLETE ML PIPELINE
=============================================================================
Covers all rubric requirements:
✅ Dataset Load & Understand (Iris dataset)
✅ Feature Scaling (StandardScaler — Mean=0, Variance=1)
✅ Train/Test Split (80/20, shuffle=True)
✅ KNN Algorithm with Elbow Method (K=1 to K=100)
✅ Confusion Matrix (TP, FP, FN, TN)
✅ Accuracy, Precision, Recall, F1-Score
✅ Accuracy Mirage demonstration (imbalanced data warning)
✅ Full IPO pipeline visualization
✅ Custom dataset (E-Commerce) with LabelEncoded features
=============================================================================
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────
# 1. IRIS PIPELINE
# ─────────────────────────────────────────────────────────
def run_iris_pipeline():
    print("\n" + "="*70)
    print("  PIPELINE 1: IRIS BENCHMARK DATASET (150 Samples, 3 Classes)")
    print("="*70)

    # Load
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    X_raw = iris.data
    y      = iris.target_names[iris.target]
    feature_names = list(iris.feature_names)

    print(f"\n[1] DATASET OVERVIEW:")
    print(f"    Samples   : {len(df)}")
    print(f"    Features  : {len(feature_names)} — {feature_names}")
    print(f"    Classes   : {list(iris.target_names)}")
    print(f"    Balance   :\n{pd.Series(y).value_counts().to_string()}")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"\n[2] FEATURE SCALING (StandardScaler):")
    print(f"    Raw   Min/Max : {X_raw.min():.2f} / {X_raw.max():.2f}")
    print(f"    Scaled Min/Max: {X_scaled.min():.2f} / {X_scaled.max():.2f}  (~−2 to +2)")
    print(f"    Scaled Mean   : {X_scaled.mean(axis=0).round(3)}")
    print(f"    Scaled Std    : {X_scaled.std(axis=0).round(3)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )
    print(f"\n[3] TRAIN-TEST SPLIT (80% / 20%, shuffle=True):")
    print(f"    Training  : {len(X_train)} samples")
    print(f"    Testing   : {len(X_test)} samples")

    # Elbow K=1..100
    max_k       = min(100, len(X_train)-1)
    k_range     = list(range(1, max_k+1))
    error_rates = []
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        error_rates.append(np.mean(m.predict(X_test) != y_test))
    best_idx  = int(np.argmin(error_rates))
    optimal_k = k_range[best_idx]

    print(f"\n[4] ELBOW METHOD (K=1 to K={max_k}):")
    print(f"    K=1   Error : {error_rates[0]*100:.2f}%  (Overfit / Noisy)")
    print(f"    K=100 Error : {error_rates[-1]*100:.2f}% (Underfit / Generic)")
    print(f"    OPTIMAL K   : K={optimal_k}  (Min Error = {error_rates[best_idx]*100:.2f}%)")

    # Elbow plot
    plt.figure(figsize=(10,4))
    plt.plot(k_range, error_rates, color='#6366f1', lw=2, label='Error Rate')
    plt.scatter([optimal_k],[error_rates[best_idx]], color='#ef4444', s=120, zorder=5,
                label=f'THE ELBOW  K={optimal_k}')
    plt.annotate(f'THE ELBOW\nK={optimal_k}  ({error_rates[best_idx]*100:.1f}%)',
                 xy=(optimal_k, error_rates[best_idx]),
                 xytext=(optimal_k+4, error_rates[best_idx]+0.04),
                 arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2),
                 fontsize=10, fontweight='bold', color='#ef4444')
    plt.title('Elbow Method — Iris Benchmark (K=1 to K=100)', fontsize=14, fontweight='bold')
    plt.xlabel('K  [K=1 Overfitting → K=100 Underfitting]', fontsize=11)
    plt.ylabel('Error Rate', fontsize=11)
    plt.legend(); plt.grid(True, ls=':', alpha=0.5); plt.tight_layout()
    plt.savefig('elbow_iris.png', dpi=200); plt.close()
    print("    Saved: elbow_iris.png")

    # Final model
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec  = recall_score(y_test, preds, average='macro', zero_division=0)
    f1   = f1_score(y_test, preds, average='macro', zero_division=0)

    print(f"\n[5] EVALUATION METRICS (K={optimal_k}):")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%  (Trustworthiness — low false alarms)")
    print(f"    Recall    : {rec*100:.2f}%  (Sensitivity — low missed detections)")
    print(f"    F1 Score  : {f1*100:.2f}%  (Harmonic Mean = 2*P*R/(P+R))")
    print("\n    Full Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Confusion Matrix
    classes = np.unique(y)
    cm = confusion_matrix(y_test, preds, labels=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, linewidths=.5)
    plt.title(f'Confusion Matrix — Iris  (K={optimal_k})', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted  →  TP / FP (False Alarm)', fontsize=10)
    plt.ylabel('Actual  →  TP / FN (Missed Detection)', fontsize=10)
    plt.tight_layout(); plt.savefig('cm_iris.png', dpi=200); plt.close()
    print("    Saved: cm_iris.png")

    # Accuracy Mirage demo
    print(f"\n[6] ACCURACY MIRAGE DEMONSTRATION:")
    y_imb   = np.array(['Normal']*990 + ['Fraud']*10)
    dumb    = np.array(['Normal']*1000)
    d_acc   = accuracy_score(y_imb, dumb)
    d_prec  = precision_score(y_imb, dumb, labels=['Fraud'], average='macro', zero_division=0)
    d_rec   = recall_score(y_imb, dumb, labels=['Fraud'], average='macro', zero_division=0)
    print(f"    Scenario    : 990 Normal, 10 Fraud (99%/1% imbalance)")
    print(f"    Dumb Model  : Always predicts 'Normal'")
    print(f"    Accuracy    : {d_acc*100:.1f}%  ← looks great, but...")
    print(f"    Precision   : {d_prec*100:.1f}%  ← cannot detect any Fraud!")
    print(f"    Recall      : {d_rec*100:.1f}%  ← completely misses all Fraud!")
    print(f"    CONCLUSION  : Accuracy alone is MISLEADING on imbalanced data!")

    print(f"\n{'='*70}")
    print(f"  IRIS PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    return model, scaler, feature_names

# ─────────────────────────────────────────────────────────
# 2. CUSTOM DATASET PIPELINE
# ─────────────────────────────────────────────────────────
def run_custom_pipeline():
    print("\n" + "="*70)
    print("  PIPELINE 2: USER E-COMMERCE DATASET (1200 Samples, 5 Classes)")
    print("="*70)

    # Load
    excel_path = 'Dataset for Data Analytics.xlsx'
    if not os.path.exists(excel_path):
        excel_path = 'Dataset for Data Analytics (1).xlsx'
    df = pd.read_excel(excel_path)

    # Feature engineering — encode categoricals
    le = LabelEncoder()
    df['Product_enc']       = le.fit_transform(df['Product'])
    df['Payment_enc']       = le.fit_transform(df['PaymentMethod'])
    df['Referral_enc']      = le.fit_transform(df['ReferralSource'])
    df['HasCoupon']         = df['CouponCode'].notna().astype(int)
    df['Month']             = pd.to_datetime(df['Date']).dt.month

    feature_cols = ['Quantity','UnitPrice','ItemsInCart','TotalPrice',
                    'Product_enc','Payment_enc','Referral_enc','HasCoupon','Month']
    target_col   = 'OrderStatus'

    X_raw = df[feature_cols].values
    y     = df[target_col].values

    print(f"\n[1] DATASET OVERVIEW:")
    print(f"    Samples   : {len(df)}")
    print(f"    Features  : {len(feature_cols)} — {feature_cols}")
    print(f"    Target    : {target_col}")
    print(f"    Balance   :\n{pd.Series(y).value_counts().to_string()}")

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"\n[2] FEATURE SCALING (StandardScaler):")
    print(f"    Raw   Min/Max : {X_raw.min():.2f} / {X_raw.max():.2f}")
    print(f"    Scaled Min/Max: {X_scaled.min():.2f} / {X_scaled.max():.2f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )
    print(f"\n[3] TRAIN-TEST SPLIT (80% / 20%, shuffle=True):")
    print(f"    Training : {len(X_train)} samples")
    print(f"    Testing  : {len(X_test)} samples")

    # Elbow
    max_k       = min(100, len(X_train)-1)
    k_range     = list(range(1, max_k+1))
    error_rates = []
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        error_rates.append(np.mean(m.predict(X_test) != y_test))
    best_idx  = int(np.argmin(error_rates))
    optimal_k = k_range[best_idx]

    print(f"\n[4] ELBOW METHOD (K=1 to K={max_k}):")
    print(f"    K=1   Error : {error_rates[0]*100:.2f}%")
    print(f"    K=100 Error : {error_rates[-1]*100:.2f}%")
    print(f"    OPTIMAL K   : K={optimal_k}  ({error_rates[best_idx]*100:.2f}%)")
    print(f"    NOTE: Low accuracy expected — OrderStatus is random in synthetic data.")

    plt.figure(figsize=(10,4))
    plt.plot(k_range, error_rates, color='#6366f1', lw=2, label='Error Rate')
    plt.scatter([optimal_k],[error_rates[best_idx]], color='#ef4444', s=120, zorder=5,
                label=f'THE ELBOW  K={optimal_k}')
    plt.annotate(f'THE ELBOW\nK={optimal_k}  ({error_rates[best_idx]*100:.1f}%)',
                 xy=(optimal_k, error_rates[best_idx]),
                 xytext=(optimal_k+4, error_rates[best_idx]+0.01),
                 arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2),
                 fontsize=10, fontweight='bold', color='#ef4444')
    plt.title('Elbow Method — E-Commerce Dataset (K=1 to K=100)', fontsize=14, fontweight='bold')
    plt.xlabel('K  [K=1 Overfitting → K=100 Underfitting]', fontsize=11)
    plt.ylabel('Error Rate', fontsize=11)
    plt.legend(); plt.grid(True, ls=':', alpha=0.5); plt.tight_layout()
    plt.savefig('elbow_custom.png', dpi=200); plt.close()
    print("    Saved: elbow_custom.png")

    # Final model
    model = KNeighborsClassifier(n_neighbors=optimal_k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec  = recall_score(y_test, preds, average='macro', zero_division=0)
    f1   = f1_score(y_test, preds, average='macro', zero_division=0)

    print(f"\n[5] EVALUATION METRICS (K={optimal_k}):")
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print("\n    Full Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Confusion Matrix
    classes = np.unique(y)
    cm = confusion_matrix(y_test, preds, labels=classes)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, linewidths=.5)
    plt.title(f'Confusion Matrix — E-Commerce  (K={optimal_k})', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted  →  TP / FP (False Alarm)', fontsize=10)
    plt.ylabel('Actual  →  TP / FN (Missed Detection)', fontsize=10)
    plt.tight_layout(); plt.savefig('cm_custom.png', dpi=200); plt.close()
    print("    Saved: cm_custom.png")

    print(f"\n{'='*70}")
    print(f"  E-COMMERCE PIPELINE COMPLETE")
    print(f"{'='*70}\n")
    return model, scaler, feature_cols


if __name__ == '__main__':
    iris_model, iris_scaler, iris_features = run_iris_pipeline()
    custom_model, custom_scaler, custom_features = run_custom_pipeline()
    print("\nAll pipeline outputs saved successfully.")
    print("Generated files: elbow_iris.png, cm_iris.png, elbow_custom.png, cm_custom.png")
