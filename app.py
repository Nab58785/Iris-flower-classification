"""
Project 2: Data Classification Using AI — Streamlit Dashboard
Covers all rubric requirements: Dataset Load, Scaling, Split, KNN, Elbow,
Confusion Matrix (TP/FP/FN/TN), Precision/Recall/F1, Accuracy Mirage, Full Pipeline.
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Project 2 — AI Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero {
    background: linear-gradient(135deg,#1e3a8a,#4f46e5,#7c3aed);
    padding: 2rem; border-radius: 14px; color: #fff; text-align: center;
    box-shadow: 0 8px 24px rgba(79,70,229,.35); margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .3rem; }
.hero p  { opacity: .9; margin: 0; font-size: 1rem; }
.kpi { background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.25);
       border-radius: 10px; padding: 1rem; text-align: center; }
.kpi .val { font-size: 1.8rem; font-weight: 800; color: #4f46e5; }
.kpi .lbl { font-size: .8rem; color: #6b7280; margin-top: .2rem; }
.cm-box { border-radius: 8px; padding: 12px; text-align: center; }
.tp { background: rgba(16,185,129,.15); border: 2px solid #10b981; }
.fp { background: rgba(245,158,11,.15); border: 2px solid #f59e0b; }
.fn { background: rgba(239,68,68,.15);  border: 2px solid #ef4444; }
.tn { background: rgba(59,130,246,.15); border: 2px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
ds_choice = st.sidebar.radio("Dataset:", ["🌸 Iris Benchmark", "📦 E-Commerce Dataset"])
k_val     = st.sidebar.slider("K (Neighbors):", 1, 50, 5)
test_size = st.sidebar.slider("Test Split:", 0.10, 0.40, 0.20, 0.05)
use_scale = st.sidebar.selectbox("Scaling:", ["StandardScaler", "Raw (Unscaled)"])

# ── Data loader ───────────────────────────────────────────
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
    df['species'] = [iris.target_names[t] for t in iris.target]
    return df, ['sepal_length','sepal_width','petal_length','petal_width'], 'species', "Iris Benchmark"

@st.cache_data
def load_custom_data():
    path = "Dataset for Data Analytics.xlsx"
    if not os.path.exists(path):
        path = "Dataset for Data Analytics (1).xlsx"
    df = pd.read_excel(path)
    le = LabelEncoder()
    df['Product_enc']  = le.fit_transform(df['Product'])
    df['Payment_enc']  = le.fit_transform(df['PaymentMethod'])
    df['Referral_enc'] = le.fit_transform(df['ReferralSource'])
    df['HasCoupon']    = df['CouponCode'].notna().astype(int)
    df['Month']        = pd.to_datetime(df['Date']).dt.month
    feats = ['Quantity','UnitPrice','ItemsInCart','TotalPrice','Product_enc','Payment_enc','Referral_enc','HasCoupon','Month']
    return df, feats, 'OrderStatus', "E-Commerce Dataset"

if ds_choice.startswith("🌸"):
    df, feature_cols, target_col, ds_name = load_iris_data()
else:
    df, feature_cols, target_col, ds_name = load_custom_data()

# ── Preprocessing ─────────────────────────────────────────
X_raw   = df[feature_cols].values.astype(float)
y_all   = df[target_col].values
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_raw)
X_data  = X_sc if use_scale == "StandardScaler" else X_raw

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_all, test_size=test_size, random_state=42, stratify=y_all, shuffle=True
)

# ── Live model (sidebar K) ────────────────────────────────
live_model = KNeighborsClassifier(n_neighbors=k_val)
live_model.fit(X_train, y_train)
preds = live_model.predict(X_test)
acc   = accuracy_score(y_test, preds)
prec  = precision_score(y_test, preds, average='macro', zero_division=0)
rec   = recall_score(y_test, preds, average='macro', zero_division=0)
f1    = f1_score(y_test, preds, average='macro', zero_division=0)

# ── Hero ──────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🌸 Project 2: Data Classification Using AI</h1>
  <p>Active: <strong>{ds_name}</strong> &nbsp;•&nbsp; KNN Algorithm &nbsp;•&nbsp; StandardScaler &nbsp;•&nbsp; Full Diagnostic Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col, label, value in [
    (c1,"Accuracy",   f"{acc*100:.1f}%"),
    (c2,"Precision",  f"{prec*100:.1f}%"),
    (c3,"Recall",     f"{rec*100:.1f}%"),
    (c4,"F1 Score",   f"{f1*100:.1f}%"),
]:
    col.markdown(f'<div class="kpi"><div class="val">{value}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📐 Architecture & IPO",
    "📊 Dataset Overview",
    "⚙️ Scaling & Split",
    "🤖 KNN & Elbow Curve",
    "🎯 Diagnostics & Metrics",
    "🚀 Live Classifier"
])

# ── TAB 1: ARCHITECTURE ───────────────────────────────────
with tab1:
    st.header("📐 Architectural Paradigms & IPO Framework")
    a1,a2 = st.columns(2)
    with a1:
        st.subheader("Raw Data → Intelligent Decision")
        st.markdown("""
**Traditional AI Classification Pipeline:**
- **Raw Features** → measured physical/numerical inputs
- **Distance Projection** → map into N-dimensional feature space
- **KNN Decision** → majority vote among K nearest neighbors
- **Output** → automated species/status classification

**Euclidean Distance Formula:**
$$d(p,q) = \\sqrt{\\sum_{i=1}^{n}(p_i - q_i)^2}$$
""")
    with a2:
        st.subheader("Heuristics vs Supervised Learning")
        st.table(pd.DataFrame({
            "Dimension":           ["Logic","Rules","Scalability","Noise handling"],
            "Old Way (Heuristics)":["Expert IF-ELSE","Manual coding","Fails at scale","Brittle"],
            "New Way (KNN/ML)":    ["Learned from data","fit() automatically","N dimensions","Probabilistic vote"]
        }))

    st.divider()
    st.subheader("🔄 IPO Framework")
    b1,b2,b3 = st.columns(3)
    b1.info("📥 **INPUT**\n- Load Dataset\n- Extract Features\n- StandardScaler (μ=0, σ=1)")
    b2.success("⚙️ **PROCESS**\n- Shuffle & 80/20 Split\n- Elbow Method → Optimal K\n- KNN fit() & predict()")
    b3.warning("📤 **OUTPUT**\n- Class Predictions\n- Confusion Matrix\n- Precision, Recall, F1")

# ── TAB 2: DATASET OVERVIEW ───────────────────────────────
with tab2:
    st.header(f"📊 Dataset Overview — {ds_name}")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Samples",    len(df))
    m2.metric("Classes",    df[target_col].nunique())
    m3.metric("Features",   len(feature_cols))
    m4.metric("Missing",    df[feature_cols].isna().sum().sum())

    d1,d2 = st.columns(2)
    with d1:
        st.subheader("Data Preview")
        st.dataframe(df[[*feature_cols, target_col]].head(50), use_container_width=True, height=320)
    with d2:
        st.subheader("Class Balance")
        counts = df[target_col].value_counts().reset_index()
        counts.columns = ["Class","Count"]
        if PLOTLY:
            st.plotly_chart(px.bar(counts, x="Class", y="Count", color="Class",
                                   title="Samples per Class"), use_container_width=True)
        else:
            fig,ax = plt.subplots(figsize=(5,3))
            sns.barplot(data=counts, x="Class", y="Count", ax=ax, palette="viridis")
            st.pyplot(fig)

# ── TAB 3: SCALING & SPLIT ────────────────────────────────
with tab3:
    st.header("⚙️ Feature Scaling & Train-Test Split")
    st.subheader("The Gatekeeper Rule: StandardScaler")
    st.markdown("""
**Formula:** $X_{scaled} = \\dfrac{X - \\mu}{\\sigma}$

Without scaling, a feature with range 0–3000 dominates Euclidean distance over a feature with range 0–5.
After scaling both have **Mean = 0** and **Std = 1**.
""")
    s1,s2 = st.columns(2)
    with s1:
        st.markdown("#### 🔴 Raw (Unscaled)")
        raw_df = pd.DataFrame(X_raw, columns=feature_cols)
        if PLOTLY:
            st.plotly_chart(px.box(raw_df, title=f"Raw  (max={X_raw.max():.1f})"), use_container_width=True)
        else:
            fig,ax = plt.subplots(figsize=(5,3)); sns.boxplot(data=raw_df,ax=ax); st.pyplot(fig)
    with s2:
        st.markdown("#### 🟢 Standard Scaled (Mean=0, Std=1)")
        sc_df = pd.DataFrame(X_sc, columns=feature_cols)
        if PLOTLY:
            st.plotly_chart(px.box(sc_df, title="Scaled  (−2 to +2)"), use_container_width=True)
        else:
            fig,ax = plt.subplots(figsize=(5,3)); sns.boxplot(data=sc_df,ax=ax); st.pyplot(fig)

    st.divider()
    st.subheader("Structural Integrity: Train-Test Split + Shuffle")
    sp1,sp2 = st.columns(2)
    sp1.success(f"**Training Set ({int((1-test_size)*100)}%):** {len(X_train)} samples — used for model.fit()")
    sp2.warning(f"**Testing Set ({int(test_size*100)}%):** {len(X_test)} samples — reserved for evaluation")
    st.info("**Why shuffle?** Without shuffling, Iris test set may contain only Virginica (last 30 rows), causing severe order bias. `shuffle=True` guarantees balanced splits.")

# ── TAB 4: KNN & ELBOW ────────────────────────────────────
with tab4:
    st.header("🤖 KNN Algorithm & Elbow Method (K=1 to K=100)")
    e1,e2 = st.columns(2)
    with e1:
        st.subheader("Proximity Principle")
        st.markdown("""
- **K=1** → Memorises every training point → **Overfitting / Noisy**
- **K=100** → Too smooth, ignores local structure → **Underfitting / Generic**
- **Optimal K (Elbow)** → Minimum error rate on test set
""")
        st.code("""
model = KNeighborsClassifier(n_neighbors=5)  # Instantiate
model.fit(X_train, y_train)                  # Fit
preds = model.predict(X_test)               # Predict
""", language="python")
    with e2:
        st.subheader("Majority Vote Example")
        st.markdown("""
| Neighbor | Distance | Class |
|----------|----------|-------|
| 1 | 0.12 | Setosa |
| 2 | 0.18 | Setosa |
| 3 | 0.25 | Versicolor |

K=3 → **Setosa wins** (2 votes vs 1)
""")

    st.divider()
    # Compute elbow
    max_k_elbow = min(100, len(X_train)-1)
    k_range, err_rates = [], []
    for k in range(1, max_k_elbow+1):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        err_rates.append(np.mean(m.predict(X_test) != y_test))
        k_range.append(k)
    best_idx = int(np.argmin(err_rates))
    opt_k    = k_range[best_idx]

    st.subheader(f"Elbow Curve — Optimal K = **{opt_k}**")
    if PLOTLY:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=k_range, y=err_rates, mode='lines',
                                   line=dict(color='#6366f1', width=3), name='Error Rate'))
        fig_e.add_trace(go.Scatter(x=[opt_k], y=[err_rates[best_idx]], mode='markers+text',
                                   marker=dict(size=14, color='#ef4444', symbol='diamond'),
                                   text=[f"  THE ELBOW (K={opt_k})"], textposition='top right',
                                   name='Optimal K'))
        fig_e.update_layout(
            xaxis_title="K  [1 = Overfit → 100 = Underfit]",
            yaxis_title="Error Rate",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig_e, use_container_width=True)
    else:
        fig,ax = plt.subplots(figsize=(9,4))
        ax.plot(k_range, err_rates, color='#6366f1', lw=2)
        ax.scatter([opt_k],[err_rates[best_idx]], color='#ef4444', s=120, zorder=5, label=f'Elbow K={opt_k}')
        ax.set_xlabel("K"); ax.set_ylabel("Error Rate"); ax.legend(); ax.grid(ls=':', alpha=.5)
        st.pyplot(fig)

# ── TAB 5: DIAGNOSTICS ────────────────────────────────────
with tab5:
    st.header("🎯 Model Diagnostics & Strategic Metrics")

    st.subheader("Confusion Matrix")
    classes = np.unique(y_all)
    cm      = confusion_matrix(y_test, preds, labels=classes)

    g1,g2 = st.columns([1,1])
    with g1:
        if PLOTLY:
            fig_cm = px.imshow(cm, x=classes, y=classes, text_auto=True,
                               color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual"),
                               title=f"Confusion Matrix  (K={k_val})")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            fig,ax = plt.subplots(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)

    with g2:
        st.markdown("#### Four-Box Breakdown")
        r1,r2 = st.columns(2)
        r1.markdown('<div class="cm-box tp"><b>🟢 TP — True Positive</b><br>Correctly identified as positive.</div>', unsafe_allow_html=True)
        r2.markdown('<div class="cm-box fp"><b>🟠 FP — False Positive</b><br><i>False Alarm:</i> negative predicted as positive.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        r3,r4 = st.columns(2)
        r3.markdown('<div class="cm-box fn"><b>🔴 FN — False Negative</b><br><i>Missed Detection:</i> positive predicted as negative.</div>', unsafe_allow_html=True)
        r4.markdown('<div class="cm-box tn"><b>🔵 TN — True Negative</b><br>Correctly identified as negative.</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Precision vs Recall vs F1 — Strategic Trade-offs")
    t1,t2 = st.columns(2)
    with t1:
        st.markdown("""
| Metric | Formula | Priority Use-Case |
|--------|---------|-------------------|
| **Precision** | TP / (TP+FP) | Spam filters, fraud (↓ false alarms) |
| **Recall** | TP / (TP+FN) | Medical diagnosis (↓ missed cases) |
| **F1 Score** | 2·P·R / (P+R) | Balanced measure when both matter |
""")
    with t2:
        if PLOTLY:
            bar_df = pd.DataFrame({"Metric":["Accuracy","Precision","Recall","F1"],
                                   "Score":[acc,prec,rec,f1]})
            fig_bar = px.bar(bar_df, x="Metric", y="Score", color="Metric",
                             range_y=[0,1], title="Current Model Metric Scores")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            fig,ax = plt.subplots(figsize=(5,3))
            ax.bar(["Acc","Prec","Rec","F1"],[acc,prec,rec,f1], color=["#6366f1","#10b981","#f59e0b","#ef4444"])
            ax.set_ylim(0,1); st.pyplot(fig)

    st.divider()
    st.subheader("⚠️ Accuracy Mirage — Output Validation")
    st.error("**'In imbalanced data, accuracy is a lie!'**")
    m1,m2 = st.columns(2)
    m1.markdown("""
**Scenario:** 990 Normal orders, 10 Fraud orders (99%/1%)

A dumb model that always predicts *"Normal"*:
- **Accuracy = 99.0%** ← looks amazing!
- **Precision (Fraud) = 0%** ← detects nothing
- **Recall (Fraud) = 0%** ← misses every fraud
- **F1 = ~0%** ← exposes the complete failure

**Conclusion:** Always evaluate Precision, Recall & F1 — never just Accuracy alone.
""")
    m2.markdown("""
**Why this matters:**

| Context | Wrong metric causes |
|---------|---------------------|
| Medical screening | Miss cancer patients (low Recall) |
| Spam filters | Block valid emails (low Precision) |
| Fraud detection | Miss fraud (low Recall) |
| Balanced datasets | F1 = Accuracy (OK) |

> Use **F1 Score** when classes are imbalanced.
""")

# ── TAB 6: LIVE CLASSIFIER ────────────────────────────────
with tab6:
    st.header("🚀 Live AI Classifier & Full Pipeline Architecture")
    st.subheader("Interactive Real-Time Predictor")
    st.write(f"Move sliders to change inputs → model predicts **{ds_name}** class instantly:")

    inp_cols  = st.columns(min(len(feature_cols), 4))
    pred_vals = []
    for i, col_name in enumerate(feature_cols):
        col = inp_cols[i % 4]
        lo, hi, mu = float(df[col_name].min()), float(df[col_name].max()), float(df[col_name].mean())
        pred_vals.append(col.slider(col_name, lo, hi, mu, key=f"sl_{col_name}"))

    sample   = np.array([pred_vals])
    sample_s = scaler.transform(sample) if use_scale == "StandardScaler" else sample
    pred_cls = live_model.predict(sample_s)[0]
    pred_prb = live_model.predict_proba(sample_s)[0]

    st.success(f"### 🎉 Predicted Class: **{str(pred_cls).upper()}**")
    prob_df = pd.DataFrame({"Class": classes, "Probability": pred_prb*100})
    if PLOTLY:
        st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", color="Class",
                               title="Confidence Scores (%)", height=260), use_container_width=True)
    else:
        fig,ax = plt.subplots(figsize=(6,2.5))
        sns.barplot(data=prob_df, x="Class", y="Probability", ax=ax, palette="viridis")
        st.pyplot(fig)

    st.divider()
    st.subheader("🏗️ Full Pipeline Architecture (Slide 17)")
    st.markdown("""
```
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│     INPUT STAGE      │ ─► │    PROCESS STAGE     │ ─► │    OUTPUT STAGE      │
│                      │    │                      │    │                      │
│  • Load Dataset      │    │  • Shuffle Data      │    │  • Predictions       │
│  • 4–9 Features      │    │  • 80/20 Split       │    │  • Confusion Matrix  │
│  • StandardScaler    │    │  • Elbow K=1..100    │    │  • Accuracy Mirage   │
│    (Mean=0, Var=1)   │    │  • KNN fit/predict   │    │  • Precision/Recall  │
└──────────────────────┘    └──────────────────────┘    │  • F1 Score          │
                                                         └──────────────────────┘
```
""")
    st.subheader("🌅 Emerging Horizons: Tabular → Computer Vision")
    st.info("""
**Project 2 (This project):** Tabular 4-dimensional feature vectors → KNN distance proximity classification.

**Next Evolution:** Image pixel grids (millions of dimensions) → **Deep Convolutional Neural Networks (CNNs)**
that automatically learn spatial feature hierarchies for object recognition, medical imaging, and beyond.
""")
    st.balloons()
