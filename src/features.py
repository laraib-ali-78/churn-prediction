"""Feature engineering and preprocessing pipeline module."""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET


ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add business domain engineered features to dataset:
    - tenure_group: categorical bin ('0-12', '13-24', '25-48', '49+')
    - avg_charge_per_month: TotalCharges / max(tenure, 1)
    - num_services: count of add-on services marked 'Yes'
    - has_internet: 1 if InternetService != 'No', else 0
    - is_month_to_month: 1 if Contract == 'Month-to-month', else 0

    Args:
        data: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features added.
    """
    df = data.copy()

    # 1. tenure_group (0-12, 13-24, 25-48, 49+)
    tenure_val = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["tenure_group"] = pd.cut(
        tenure_val,
        bins=[-1, 12, 24, 48, 1000],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)

    # 2. avg_charge_per_month (TotalCharges / max(tenure, 1))
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    effective_tenure = np.maximum(tenure_val, 1)
    df["avg_charge_per_month"] = total_charges / effective_tenure

    # 3. num_services (count of add-on services = 'Yes')
    addon_cols = [col for col in ADDON_SERVICES if col in df.columns]
    if addon_cols:
        df["num_services"] = (df[addon_cols] == "Yes").sum(axis=1)
    else:
        df["num_services"] = 0

    # 4. has_internet (1 if InternetService is not 'No', else 0)
    if "InternetService" in df.columns:
        df["has_internet"] = (df["InternetService"].astype(str).str.lower() != "no").astype(int)
    else:
        df["has_internet"] = 0

    # 5. is_month_to_month (1 if Contract == 'Month-to-month', else 0)
    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    else:
        df["is_month_to_month"] = 0

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for custom feature engineering."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_engineered_features(X)


def get_column_types(sample_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Determine numeric and categorical column lists for ColumnTransformer.

    Args:
        sample_df: DataFrame after feature engineering (excluding target).

    Returns:
        Tuple[List[str], List[str]]: numeric_features, categorical_features
    """
    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "avg_charge_per_month",
        "num_services",
        "SeniorCitizen",
        "has_internet",
        "is_month_to_month",
    ]
    # Filter to only existing columns
    numeric_features = [c for c in numeric_features if c in sample_df.columns]

    categorical_features = [
        c for c in sample_df.columns if c not in numeric_features and c != TARGET
    ]

    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer:
    - numeric: SimpleImputer(median) + StandardScaler
    - categorical: OneHotEncoder(handle_unknown='ignore')

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_preprocessing_pipeline() -> Tuple[Pipeline, List[str], List[str]]:
    """Create a unified preprocessing pipeline that applies FeatureEngineer followed by ColumnTransformer.

    Returns:
        Tuple[Pipeline, List[str], List[str]]: Full preprocessing pipeline, num cols, cat cols
    """
    # Create sample dummy to extract column definitions
    dummy_df = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }])
    engineered_dummy = add_engineered_features(dummy_df)
    numeric_cols, categorical_cols = get_column_types(engineered_dummy)

    col_transformer = build_preprocessor(numeric_cols, categorical_cols)

    prep_pipeline = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("column_transform", col_transformer),
        ]
    )
    return prep_pipeline, numeric_cols, categorical_cols


def prepare_data_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split train/test stratified on target, random_state=42.

    Args:
        df: Cleaned DataFrame containing features and TARGET.
        test_size: Proportion of test split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data import load_data, clean_data

    raw = load_data()
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = prepare_data_split(clean)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train churn rate: {y_train.mean():.4f}, y_test churn rate: {y_test.mean():.4f}")

    pipeline, num_cols, cat_cols = build_full_preprocessing_pipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    print(f"Transformed train shape: {X_train_trans.shape}, test shape: {X_test_trans.shape}")
    print("Feature engineering & ColumnTransformer successfully tested!")
