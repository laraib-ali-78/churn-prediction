"""Data loading and cleaning module for Customer Churn Prediction."""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw telecom churn dataset from CSV file.

    Args:
        file_path: Optional path to raw CSV file. Defaults to RAW_DATA_PATH.

    Returns:
        pd.DataFrame: Raw dataset.
    """
    path = file_path if file_path is not None else RAW_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    """Clean telecom churn dataset according to project rules:
    - Strip whitespace from all string columns
    - Convert TotalCharges to numeric, filling blanks (tenure 0) with 0.0
    - Drop customerID if present
    - Map Churn 'Yes'/'No' to 1/0 if Churn is in columns and drop_target is False
    - Remove duplicates

    Args:
        df: Input DataFrame.
        drop_target: If True, do not require or process target column (for inference).

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    data = df.copy()

    # Strip whitespace from string columns
    str_cols = data.select_dtypes(include=["object", "string", "str"]).columns
    for col in str_cols:
        data[col] = data[col].astype(str).str.strip()

    # Drop customerID if present
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # TotalCharges conversion: handle blanks and coercion
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        # Blank TotalCharges occurs when tenure is 0 (new customers)
        data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

    # Churn target mapping
    if TARGET in data.columns and not drop_target:
        if data[TARGET].dtype == object or str(data[TARGET].dtype) == "str":
            data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0})
        # If any invalid, drop or handle
        data = data.dropna(subset=[TARGET])
        data[TARGET] = data[TARGET].astype(int)

    # Remove duplicates
    data = data.drop_duplicates().reset_index(drop=True)

    return data


def run_data_pipeline() -> Tuple[pd.DataFrame, Path]:
    """Execute full data loading and cleaning, saving result to processed directory.

    Returns:
        Tuple[pd.DataFrame, Path]: Cleaned DataFrame and saved file path.
    """
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Clean data saved to {PROCESSED_DATA_PATH} with shape: {clean_df.shape}")
    return clean_df, PROCESSED_DATA_PATH


if __name__ == "__main__":
    run_data_pipeline()
