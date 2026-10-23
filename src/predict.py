"""Inference module for single customer and batch churn predictions with SHAP explainability."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODEL_PATH, OPTIMAL_THRESHOLD
from src.evaluate import get_feature_names

# Required fields for customer payload validation
REQUIRED_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

_CACHED_MODEL: Optional[Any] = None
_CACHED_EXPLAINER: Optional[Any] = None
_CACHED_FEATURE_NAMES: Optional[List[str]] = None


def get_model():
    """Load and cache the trained churn prediction pipeline."""
    global _CACHED_MODEL, _CACHED_EXPLAINER, _CACHED_FEATURE_NAMES
    if _CACHED_MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _CACHED_MODEL = joblib.load(MODEL_PATH)
        prep_pipe = _CACHED_MODEL.named_steps["prep"]
        clf = _CACHED_MODEL.named_steps["clf"]
        _CACHED_FEATURE_NAMES = get_feature_names(prep_pipe)
        _CACHED_EXPLAINER = shap.TreeExplainer(clf)
    return _CACHED_MODEL, _CACHED_EXPLAINER, _CACHED_FEATURE_NAMES


def validate_customer_payload(customer: dict) -> None:
    """Validate that all required input fields are present in payload."""
    missing = [f for f in REQUIRED_FIELDS if f not in customer]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")


def determine_risk_level(prob: float) -> str:
    """Categorize churn probability into risk levels."""
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def predict_churn(customer: dict, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Predict customer churn probability, category, risk level, and top 3 driving factors.

    Args:
        customer: Dictionary containing customer attribute keys and values.
        threshold: Optional custom decision threshold (defaults to OPTIMAL_THRESHOLD).

    Returns:
        dict: {
            "churn_probability": float,
            "prediction": "Churn" | "No churn",
            "risk_level": "Low" | "Medium" | "High",
            "top_factors": List[dict]
        }
    """
    validate_customer_payload(customer)
    model, explainer, feature_names = get_model()

    # Create 1-row DataFrame
    row_df = pd.DataFrame([customer])

    # Convert numeric fields
    row_df["tenure"] = pd.to_numeric(row_df["tenure"], errors="coerce").fillna(0)
    row_df["MonthlyCharges"] = pd.to_numeric(row_df["MonthlyCharges"], errors="coerce").fillna(0.0)
    row_df["TotalCharges"] = pd.to_numeric(row_df["TotalCharges"], errors="coerce").fillna(0.0)

    # Probability prediction
    prob = float(model.predict_proba(row_df)[0, 1])
    active_threshold = threshold if threshold is not None else OPTIMAL_THRESHOLD
    pred_label = "Churn" if prob >= active_threshold else "No churn"
    risk = determine_risk_level(prob)

    # SHAP explanations for top 3 factors
    prep_pipe = model.named_steps["prep"]
    trans_features = prep_pipe.transform(row_df)
    trans_df = pd.DataFrame(trans_features, columns=feature_names)

    shap_res = explainer(trans_df)
    if len(shap_res.shape) == 3:
        shap_vals = shap_res.values[0, :, 1]
    else:
        shap_vals = shap_res.values[0]

    # Rank by absolute impact
    top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_factors = []
    for idx in top_indices:
        val = float(shap_vals[idx])
        direction = "Increases churn risk" if val > 0 else "Decreases churn risk"
        top_factors.append({
            "feature": feature_names[idx],
            "impact": direction,
            "shap_value": round(val, 4),
        })

    return {
        "churn_probability": round(prob, 4),
        "prediction": pred_label,
        "risk_level": risk,
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.0,
    }

    result = predict_churn(sample_customer)
    print("Inference Result:")
    print(result)
