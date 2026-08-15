import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Risk Dashboard", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("churn_random_forest_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
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

# ---------------- UI ----------------
st.title("📊 Customer Churn Risk Dashboard")
st.caption("Random Forest model | ROC-AUC 0.845 | Recall 87% @ threshold 0.35")

mode = st.sidebar.radio("Mode", ["Single Customer Check", "Batch Upload (CSV)"])

if mode == "Single Customer Check":
    st.subheader("Check a single customer's churn risk")
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

    if st.button("Predict Churn Risk", type="primary"):
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

        color = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}[band]
        st.markdown(f"### Churn Probability: **{proba:.1%}**")
        st.markdown(f"### Risk Band: :{color}[{band}]")
        st.info(f"**Recommended Action:** {action}")

    with st.expander("Why does the model decide this way? (Top global drivers)"):
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(10)
        fig = px.bar(importances[::-1], orientation="h", labels={"value": "Importance", "index": "Feature"},
                     title="Top 10 Feature Importances")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("Batch risk scoring")
    st.write("Upload a CSV with the same columns as the Telco Customer Churn dataset (customerID and Churn columns are optional/ignored).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        if "Churn" in raw_df.columns:
            raw_df = raw_df.drop(columns=["Churn"])

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

        st.success(f"Scored {len(results)} customers")
        c1, c2, c3 = st.columns(3)
        c1.metric("High Risk", (results["risk_band"] == "High Risk").sum())
        c2.metric("Medium Risk", (results["risk_band"] == "Medium Risk").sum())
        c3.metric("Low Risk", (results["risk_band"] == "Low Risk").sum())

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_pie = px.pie(results, names="risk_band", title="Risk Band Distribution",
                              color="risk_band",
                              color_discrete_map={"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#2ecc71"})
            st.plotly_chart(fig_pie, use_container_width=True)
        with chart_col2:
            fig_hist = px.histogram(results, x="churn_probability", nbins=30,
                                     title="Churn Probability Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored results as CSV", csv, "churn_risk_results.csv", "text/csv")
