"""Streamlit Web Application for Customer Churn Prediction and Explainability."""

import json
from pathlib import Path
import sys
from typing import Dict, Any, List

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    OPTIMAL_THRESHOLD,
    REPORTS_DIR,
)
from src.evaluate import get_feature_names
from src.predict import (
    REQUIRED_FIELDS,
    determine_risk_level,
    predict_churn,
)

# Page configuration
st.set_page_config(
    page_title="Telco Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_inference_artifacts():
    """Load and cache the trained pipeline and SHAP explainer."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    prep_pipe = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    feature_names = get_feature_names(prep_pipe)
    explainer = shap.TreeExplainer(clf)
    return model, explainer, feature_names


@st.cache_data
def load_metrics():
    """Load evaluation metrics JSON."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# Load artifacts
try:
    model, explainer, feature_names = load_inference_artifacts()
    metrics_data = load_metrics()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Header banner
st.title("📡 Telecom Customer Churn Prediction & Explainability")
st.markdown(
    "Identify at-risk telecom customers using machine learning and explain predictions with SHAP feature attributions."
)

# ----------------- SIDEBAR INPUT FORM -----------------
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Configure customer attributes for individual scoring:")

with st.sidebar.expander("👤 Account & Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Female", "Male"], index=0)
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    partner = st.selectbox("Has Partner", ["No", "Yes"], index=0)
    dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=0)
    tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=12, step=1)

with st.sidebar.expander("🌐 Services & Subscriptions", expanded=True):
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"],
        index=0,
    )
    internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"], index=0)

    if internet_service != "No":
        online_security = st.selectbox("Online Security", ["No", "Yes"], index=0)
        online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=1)
        device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=0)
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0)
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"], index=0)
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"], index=0)
    else:
        online_security = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        tech_support = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"

with st.sidebar.expander("💳 Contract & Billing", expanded=True):
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        index=0,
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=130.0, value=75.0, step=0.5)
    default_total = round(float(tenure * monthly_charges), 2)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=max(default_total, 18.0), step=10.0)

# Build customer dictionary from inputs
current_customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs(["🎯 Predict & Batch Scoring", "📈 Model Performance", "ℹ️ About & Methodology"])

# ----------------- TAB 1: PREDICT & BATCH -----------------
with tab1:
    st.subheader("Single Customer Risk Scoring")
    col_pred_btn, col_thresh = st.columns([2, 1])
    with col_thresh:
        active_threshold = st.slider(
            "Decision Threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(OPTIMAL_THRESHOLD),
            step=0.01,
            help="Optimal threshold determined via F1-maximization on validation set.",
        )
    with col_pred_btn:
        st.write("")
        predict_clicked = st.button("🚀 Score Customer", type="primary", use_container_width=True)

    if predict_clicked:
        res = predict_churn(current_customer, threshold=active_threshold)
        prob = res["churn_probability"]
        pred = res["prediction"]
        risk = res["risk_level"]
        factors = res["top_factors"]

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.progress(prob)

        with res_col2:
            if pred == "Churn":
                st.error(f"Prediction: **{pred}**")
            else:
                st.success(f"Prediction: **{pred}**")
            st.caption(f"Decision threshold: {active_threshold:.2f}")

        with res_col3:
            if risk == "High":
                st.error(f"Risk Tier: **{risk} Risk**")
            elif risk == "Medium":
                st.warning(f"Risk Tier: **{risk} Risk**")
            else:
                st.success(f"Risk Tier: **{risk} Risk**")
            st.caption("Risk tiers: Low (<35%), Medium (35-65%), High (>65%)")

        st.markdown("#### 🔍 Key Risk Drivers (SHAP Explanations)")
        f_cols = st.columns(3)
        for i, factor in enumerate(factors):
            with f_cols[i]:
                impact_color = "red" if "Increases" in factor["impact"] else "green"
                st.markdown(
                    f"""
                    <div style="padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb; background: #fafafa;">
                        <span style="font-size: 0.85rem; color: #6b7280;">Driver #{i+1}</span>
                        <h4 style="margin: 4px 0;">{factor['feature']}</h4>
                        <p style="color: {impact_color}; font-weight: 600; margin: 0;">
                            {factor['impact']} ({factor['shap_value']:+.4f})
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("📂 Batch CSV Scoring")
    st.markdown("Upload a CSV file containing multiple customer records to score churn risk in bulk.")

    sample_csv_data = pd.DataFrame([current_customer])
    st.download_button(
        "📥 Download Sample CSV Template",
        sample_csv_data.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Customers CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** records for scoring:")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("⚡ Score Batch Records", type="secondary"):
                with st.spinner("Processing batch predictions..."):
                    # Check missing columns
                    missing_cols = [c for c in REQUIRED_FIELDS if c not in batch_df.columns]
                    if missing_cols:
                        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}")
                    else:
                        prep_pipe = model.named_steps["prep"]
                        probs = model.predict_proba(batch_df)[:, 1]
                        batch_df["churn_probability"] = np.round(probs, 4)
                        batch_df["prediction"] = np.where(probs >= active_threshold, "Churn", "No churn")
                        batch_df["risk_level"] = [determine_risk_level(p) for p in probs]

                        st.success("Batch scoring completed successfully!")

                        # Summary stats
                        b_col1, b_col2, b_col3 = st.columns(3)
                        b_col1.metric("Total Scored", len(batch_df))
                        b_col2.metric("Predicted Churners", int((batch_df["prediction"] == "Churn").sum()))
                        b_col3.metric("High Risk Tier", int((batch_df["risk_level"] == "High").sum()))

                        st.dataframe(
                            batch_df[["churn_probability", "prediction", "risk_level"] + [c for c in REQUIRED_FIELDS[:6]]].head(10),
                            use_container_width=True,
                        )

                        scored_csv = batch_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Scored Results CSV",
                            data=scored_csv,
                            file_name="scored_customers_output.csv",
                            mime="text/csv",
                        )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")

# ----------------- TAB 2: MODEL PERFORMANCE -----------------
with tab2:
    st.subheader("🏆 Held-Out Test Set Performance")
    if metrics_data:
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        kpi1.metric("ROC-AUC", f"{metrics_data.get('roc_auc', 0):.4f}")
        kpi2.metric("PR-AUC", f"{metrics_data.get('pr_auc', 0):.4f}")
        kpi3.metric("F1-Score", f"{metrics_data.get('f1', 0):.4f}")
        kpi4.metric("Recall", f"{metrics_data.get('recall', 0) * 100:.1f}%")
        kpi5.metric("Precision", f"{metrics_data.get('precision', 0) * 100:.1f}%")
        kpi6.metric("Accuracy", f"{metrics_data.get('accuracy', 0) * 100:.1f}%")

        st.caption(f"Evaluated on 1,405 held-out customer records with decision threshold = {metrics_data.get('threshold', 0.5):.2f}")

    st.markdown("#### Model Diagnostics & Curves")
    diag_col1, diag_col2 = st.columns(2)

    cm_path = FIGURES_DIR / "confusion_matrix.png"
    if cm_path.exists():
        diag_col1.image(str(cm_path), caption="Confusion Matrix on Held-Out Test Set", use_container_width=True)

    roc_path = FIGURES_DIR / "roc_curve.png"
    if roc_path.exists():
        diag_col2.image(str(roc_path), caption="ROC Curve", use_container_width=True)

    diag_col3, diag_col4 = st.columns(2)
    pr_path = FIGURES_DIR / "precision_recall_curve.png"
    if pr_path.exists():
        diag_col3.image(str(pr_path), caption="Precision-Recall Curve", use_container_width=True)

    calib_path = FIGURES_DIR / "calibration_curve.png"
    if calib_path.exists():
        diag_col4.image(str(calib_path), caption="Calibration Curve (Reliability Diagram)", use_container_width=True)

    st.markdown("#### Feature Importance & Global Explainability")
    fi_col1, fi_col2 = st.columns(2)
    fi_path = FIGURES_DIR / "feature_importance.png"
    if fi_path.exists():
        fi_col1.image(str(fi_path), caption="Random Forest Feature Importance", use_container_width=True)

    shap_beeswarm = FIGURES_DIR / "shap_summary_beeswarm.png"
    if shap_beeswarm.exists():
        fi_col2.image(str(shap_beeswarm), caption="SHAP Summary (Beeswarm) Plot", use_container_width=True)

    shap_waterfall = FIGURES_DIR / "shap_waterfall_high_risk.png"
    if shap_waterfall.exists():
        st.image(str(shap_waterfall), caption="SHAP Waterfall Plot for High-Risk Customer Instance", use_container_width=True)

# ----------------- TAB 3: ABOUT & METHODOLOGY -----------------
with tab3:
    st.subheader("📖 Project Overview & Methodology")
    st.markdown(
        """
        ### Business Context
        Customer churn represents one of the largest drains on telecom enterprise revenue. Because customer acquisition costs
        typically range from 5x to 10x the cost of retention, proactive intervention using calibrated machine learning models
        generates direct financial return.

        ### Dataset
        - **Source**: Public IBM Telco Customer Churn Dataset
        - **Volume**: 7,021 cleaned customer profiles
        - **Features**: Contract terms, internet and phone service add-ons, tenure, billing method, monthly charges, and total charges.
        - **Class Distribution**: 26.45% churn rate (imbalanced binary classification).

        ### Modeling Architecture
        1. **Feature Engineering**:
           - `tenure_group`: Quantized customer longevity bins (`0-12`, `13-24`, `25-48`, `49+` months).
           - `avg_charge_per_month`: Effective monthly cost ratio capturing billing anomalies.
           - `num_services`: Sum of optional security, backup, and entertainment add-on services.
           - `has_internet` & `is_month_to_month`: Direct high-signal interaction flags.
        2. **Pipeline Integration**:
           - Numeric scaling using `StandardScaler` with median imputation.
           - Categorical one-hot encoding (`handle_unknown='ignore'`).
           - All preprocessing fitted **strictly on the training split** to prevent data leakage.
        3. **Algorithms Compared**:
           - Logistic Regression (class-weighted)
           - Random Forest Classifier (balanced weights, hyperparameter tuned)
           - XGBoost (scale_pos_weight compensation)
        4. **Decision Threshold Tuning**:
           - Optimal probability threshold tuned via cross-validation to maximize F1-score and capture high recall on actual churners.

        ### Real-World Limitations
        - **Cross-Sectional Nature**: Static snapshot without time-series usage telemetry or support ticketing trends.
        - **Demographic Fair Lending**: Care must be taken that retention discounts are distributed without unfair demographic bias.
        """
    )
