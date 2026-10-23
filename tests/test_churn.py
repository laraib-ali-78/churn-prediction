"""Pytest test suite for Customer Churn Prediction project."""

import pytest
import pandas as pd
import numpy as np

from src.data import clean_data
from src.features import (
    build_full_preprocessing_pipeline,
    add_engineered_features,
    prepare_data_split,
)
from src.predict import predict_churn, REQUIRED_FIELDS


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Fixture providing raw uncleaned telecom records."""
    return pd.DataFrame([
        {
            "customerID": "1234-ABCD",
            "gender": "Male  ",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 0,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 45.0,
            "TotalCharges": " ",  # Blank text with tenure 0
            "Churn": "No",
        },
        {
            "customerID": "5678-EFGH",
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.5,
            "TotalCharges": "1146.0",
            "Churn": "Yes",
        },
    ])


@pytest.fixture
def valid_customer_dict() -> dict:
    """Fixture providing valid customer input for inference."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 60.0,
        "TotalCharges": 1080.0,
    }


def test_data_cleaning_works(sample_raw_data: pd.DataFrame):
    """Test data cleaning: customerID dropped, blanks converted to 0, Churn mapped to int."""
    cleaned = clean_data(sample_raw_data)

    # 1. customerID dropped
    assert "customerID" not in cleaned.columns

    # 2. TotalCharges converted to numeric and blank filled with 0.0
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert np.issubdtype(cleaned["TotalCharges"].dtype, np.number)

    # 3. Churn mapped to 0 and 1 integers
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned.loc[0, "Churn"] == 0
    assert cleaned.loc[1, "Churn"] == 1

    # 4. Whitespace stripped
    assert cleaned.loc[0, "gender"] == "Male"


def test_pipeline_fits_and_predicts(sample_raw_data: pd.DataFrame):
    """Test full preprocessing pipeline fit, transform, and prediction."""
    # Create sufficient rows for pipeline fitting
    extended_data = pd.concat([sample_raw_data] * 10, ignore_index=True)
    cleaned = clean_data(extended_data)
    X = cleaned.drop(columns=["Churn"])
    y = cleaned["Churn"]

    prep_pipe, _, _ = build_full_preprocessing_pipeline()
    transformed = prep_pipe.fit_transform(X)

    assert transformed.shape[0] == len(cleaned)
    assert transformed.shape[1] > 10
    assert not np.isnan(transformed).any()


def test_probability_between_0_and_1(valid_customer_dict: dict):
    """Test inference output probability lies strictly in [0.0, 1.0]."""
    result = predict_churn(valid_customer_dict)
    prob = result["churn_probability"]

    assert isinstance(prob, (float, np.floating))
    assert 0.0 <= prob <= 1.0


def test_missing_field_raises_value_error(valid_customer_dict: dict):
    """Test that missing required fields raise a descriptive ValueError."""
    for field in REQUIRED_FIELDS:
        incomplete_customer = valid_customer_dict.copy()
        del incomplete_customer[field]

        with pytest.raises(ValueError) as exc_info:
            predict_churn(incomplete_customer)

        assert f"Missing required field(s): {field}" in str(exc_info.value)


def test_predict_churn_returns_expected_keys(valid_customer_dict: dict):
    """Test that predict_churn returns all required keys and types."""
    result = predict_churn(valid_customer_dict)

    expected_keys = {"churn_probability", "prediction", "risk_level", "top_factors"}
    assert set(result.keys()) == expected_keys

    assert result["prediction"] in ["Churn", "No churn"]
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(result["top_factors"], list)
    assert len(result["top_factors"]) == 3

    for factor in result["top_factors"]:
        assert "feature" in factor
        assert "impact" in factor
        assert "shap_value" in factor
