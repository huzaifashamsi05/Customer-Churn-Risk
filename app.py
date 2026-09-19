import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio

st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide", page_icon="📊")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg-0: #05070d;
    --bg-1: #0a0f1c;
    --accent-1: #34d399;
    --accent-2: #38bdf8;
    --danger: #ef4444;
    --warn: #f59e0b;
    --ok: #22c55e;
    --text-hi: #f2f6ff;
    --text-lo: #93a2c2;
    --border: rgba(148, 163, 184, 0.14);
}

html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(52,211,153,0.10) 0%, transparent 42%),
        radial-gradient(circle at 90% 10%, rgba(56,189,248,0.10) 0%, transparent 40%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-0) 100%);
}
#MainMenu, footer, header { visibility: hidden; }

.hero-wrap {
    padding: 2.4rem 2.2rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(52,211,153,0.10), rgba(56,189,248,0.08));
    border: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.hero-eyebrow {
    display: inline-block; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--accent-1); font-weight: 600;
    background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.28);
    padding: 0.3rem 0.75rem; border-radius: 999px; margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em;
    color: var(--text-hi); margin: 0 0 0.5rem 0; line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--text-lo); font-size: 1.0rem; max-width: 640px; margin: 0; }
.hero-badges { margin-top: 1.1rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero-badge {
    font-size: 0.78rem; color: var(--text-hi);
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    padding: 0.35rem 0.8rem; border-radius: 999px;
}

.section-label {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent-2); font-weight: 700; margin: 1.2rem 0 0.4rem 0;
}
.section-title { font-size: 1.3rem; font-weight: 700; color: var(--text-hi); margin: 0 0 1rem 0; }

section[data-testid="stSidebar"] { background: var(--bg-0); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stRadio label { color: var(--text-hi) !important; font-weight: 600; }

.stButton>button {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    color: #04140f; font-weight: 700; border: none; border-radius: 10px;
    padding: 0.65rem 1.8rem;
    box-shadow: 0 8px 24px -8px rgba(52,211,153,0.5);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 28px -6px rgba(52,211,153,0.65); }

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--text-lo) !important; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }

[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: rgba(255,255,255,0.02); }

p, .stMarkdown, label { color: var(--text-lo); }
h1, h2, h3 { color: var(--text-hi); }

.result-card {
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    border: 1px solid var(--border);
    margin: 0.6rem 0 1.2rem 0;
}
.result-card.high { background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(239,68,68,0.03)); border-color: rgba(239,68,68,0.35); }
.result-card.medium { background: linear-gradient(135deg, rgba(245,158,11,0.14), rgba(245,158,11,0.03)); border-color: rgba(245,158,11,0.35); }
.result-card.low { background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(34,197,94,0.03)); border-color: rgba(34,197,94,0.35); }
.result-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
.result-prob { font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; font-size: 2.4rem; font-weight: 800; color: var(--text-hi); }

/* Sidebar mode selector - card style */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.6rem;
    display: flex;
    flex-direction: column;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin: 0 !important;
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    border-color: rgba(52,211,153,0.4);
    transform: translateX(2px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    border-color: transparent;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #04140f !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: var(--text-hi);
    font-size: 0.92rem;
    font-weight: 600;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] .stRadio > label {
    display: none;
}
.result-band {
    display: inline-block; font-weight: 700; font-size: 0.85rem;
    padding: 0.35rem 0.9rem; border-radius: 999px; margin-top: 0.3rem;
}
.result-band.high { background: rgba(239,68,68,0.2); color: #fca5a5; }
.result-band.medium { background: rgba(245,158,11,0.2); color: #fcd34d; }
.result-band.low { background: rgba(34,197,94,0.2); color: #86efac; }
.result-action {
    margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.08);
    color: var(--text-hi); font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

pio.templates["churn_dark"] = pio.templates["plotly_dark"]
pio.templates["churn_dark"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9d4ea"),
    colorway=["#34d399", "#38bdf8", "#f59e0b", "#ef4444", "#a78bfa"],
)
pio.templates.default = "churn_dark"


@st.cache_resource
def load_model():
    try:
        model = joblib.load("churn_random_forest_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
    except Exception as e:
        st.error(f"Could not load the trained model files: {e}")
        st.stop()
    return model, feature_columns


model, feature_columns = load_model()

CHOSEN_THRESHOLD = 0.35


def risk_band(p):
    if p >= 0.6:
        return "High Risk"
    elif p >= CHOSEN_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


def recommend_action(is_month_to_month, tenure, band):
    if is_month_to_month and band != "Low Risk":
        return "Offer discounted 1-year contract upgrade"
    elif tenure < 6 and band == "High Risk":
        return "Proactive onboarding call + loyalty credit"
    elif band == "High Risk":
        return "Priority retention call with personalized offer"
    elif band == "Medium Risk":
        return "Send automated retention email with plan review"
    else:
        return "No action needed"


def preprocess_single(raw_dict):
    df = pd.DataFrame([raw_dict])
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"], include_lowest=True
    )
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    binary_cols = [c for c in df.columns if df[c].dropna().isin(["Yes", "No"]).all()]
    for col in binary_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    categorical_cols = df.select_dtypes(include="object").columns.tolist() + ["tenure_group"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def preprocess_batch(raw_df):
    df = raw_df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"], include_lowest=True
    )
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    binary_cols = [c for c in df.columns if df[c].dropna().isin(["Yes", "No"]).all()]
    for col in binary_cols:
        df[col] = df[col].map({"No": 0, "Yes": 1})
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    categorical_cols = df.select_dtypes(include="object").columns.tolist() + ["tenure_group"]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    return df_encoded[feature_columns], df


# ---------------- HERO ----------------
st.markdown("""
<div class="hero-wrap">
    <span class="hero-eyebrow">Machine Learning · Random Forest</span>
    <h1 class="hero-title">📊 Customer Churn <span>Risk Dashboard</span></h1>
    <p class="hero-sub">Predict which customers are likely to leave, understand why, and get a
    recommended retention action — one customer at a time or in bulk.</p>
    <div class="hero-badges">
        <span class="hero-badge">🎯 ROC-AUC 0.845</span>
        <span class="hero-badge">📈 Recall 87% @ 0.35</span>
        <span class="hero-badge">⚙️ Explainable feature importances</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="section-label" style="margin-top:0;">Navigate</p>', unsafe_allow_html=True)
mode = st.sidebar.radio(
    "Mode",
    ["👤  Single Customer Check", "📁  Batch Upload (CSV)"],
    label_visibility="collapsed",
)
mode = "Single Customer Check" if "Single" in mode else "Batch Upload (CSV)"

band_class = {"High Risk": "high", "Medium Risk": "medium", "Low Risk": "low"}

if mode == "Single Customer Check":
    st.markdown('<p class="section-label">Step 1</p><p class="section-title">Enter customer details</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Has Partner", ["No", "Yes"])
        Dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with col3:
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)

    TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, MonthlyCharges * tenure)

    st.write("")
    if st.button("🔍 Predict Churn Risk", type="primary"):
        raw = dict(
            gender=gender, SeniorCitizen=1 if SeniorCitizen == "Yes" else 0,
            Partner=Partner, Dependents=Dependents, tenure=tenure,
            PhoneService=PhoneService, MultipleLines=MultipleLines,
            InternetService=InternetService, OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup, DeviceProtection=DeviceProtection,
            TechSupport=TechSupport, StreamingTV=StreamingTV, StreamingMovies=StreamingMovies,
            Contract=Contract, PaperlessBilling=PaperlessBilling, PaymentMethod=PaymentMethod,
            MonthlyCharges=MonthlyCharges, TotalCharges=TotalCharges
        )
        X_input = preprocess_single(raw)
        proba = model.predict_proba(X_input)[0][1]
        band = risk_band(proba)
        action = recommend_action(Contract == "Month-to-month", tenure, band)
        cls = band_class[band]

        st.markdown('<p class="section-label">Result</p><p class="section-title">Prediction</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="result-row">
                <div>
                    <div class="result-prob">{proba:.1%}</div>
                    <div style="color:var(--text-lo); font-size:0.85rem;">Churn probability</div>
                </div>
                <div style="text-align:right;">
                    <span class="result-band {cls}">{band}</span>
                </div>
            </div>
            <div class="result-action">🎯 <b>Recommended action:</b> {action}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("💡 Why does the model decide this way? (Top global drivers)"):
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(10)
        fig = px.bar(importances[::-1], orientation="h", labels={"value": "Importance", "index": "Feature"},
                     title="Top 10 Feature Importances")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown('<p class="section-label">Step 1</p><p class="section-title">Upload customer data</p>', unsafe_allow_html=True)
    st.write("Upload a CSV with the same columns as the Telco Customer Churn dataset (customerID and Churn columns are optional/ignored).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read this file as a CSV: {e}")
            st.stop()
        if "Churn" in raw_df.columns:
            raw_df = raw_df.drop(columns=["Churn"])
        required_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
        missing_cols = [c for c in required_columns if c not in raw_df.columns]
        if missing_cols:
            st.error(f"This CSV is missing required column(s): {', '.join(missing_cols)}")
            st.stop()

        with st.spinner("Scoring customers..."):
            X_batch, original_df = preprocess_batch(raw_df)
            probas = model.predict_proba(X_batch)[:, 1]
            bands = [risk_band(p) for p in probas]
            results = original_df.copy()
            results["churn_probability"] = probas
            results["risk_band"] = bands
            results["recommended_action"] = [
                recommend_action(row.get("Contract") == "Month-to-month", row.get("tenure", 99), band)
                for (_, row), band in zip(original_df.iterrows(), bands)
            ]
            results = results.sort_values("churn_probability", ascending=False)

        st.success(f"✅ Scored {len(results)} customers")

        st.markdown('<p class="section-label">Summary</p><p class="section-title">Risk breakdown</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("High Risk", int((results["risk_band"] == "High Risk").sum()))
        c2.metric("Medium Risk", int((results["risk_band"] == "Medium Risk").sum()))
        c3.metric("Low Risk", int((results["risk_band"] == "Low Risk").sum()))

        st.write("")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_pie = px.pie(results, names="risk_band", title="Risk Band Distribution",
                              color="risk_band", hole=0.45,
                              color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"})
            fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        with chart_col2:
            fig_hist = px.histogram(results, x="churn_probability", nbins=30,
                                     title="Churn Probability Distribution",
                                     color_discrete_sequence=["#38bdf8"])
            fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<p class="section-label">Detail</p><p class="section-title">Scored customers</p>', unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download scored results as CSV", csv, "churn_risk_results.csv", "text/csv")
