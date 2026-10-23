"""Model training, hyperparameter tuning, and threshold selection pipeline."""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import sklearn

from src.config import (
    MODEL_PATH,
    MODEL_CARD_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    PROCESSED_DATA_PATH,
)
from src.data import load_data, clean_data
from src.features import (
    build_full_preprocessing_pipeline,
    prepare_data_split,
)


def tune_decision_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float, float, float]:
    """Find the probability threshold that maximizes F1 score on validation set.

    Args:
        y_true: True binary target labels.
        y_proba: Predicted probabilities for the positive class (Churn = 1).

    Returns:
        Tuple[float, float, float, float]: (best_threshold, best_f1, precision_at_best, recall_at_best)
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            best_prec = precision_score(y_true, preds, zero_division=0)
            best_rec = recall_score(y_true, preds, zero_division=0)

    return float(best_thresh), float(best_f1), float(best_prec), float(best_rec)


def update_config_threshold(new_threshold: float) -> None:
    """Persist optimal threshold in src/config.py."""
    config_file = Path(__file__).resolve().parent / "config.py"
    with open(config_file, "r") as f:
        content = f.read()

    lines = content.splitlines()
    updated_lines = []
    found = False
    for line in lines:
        if line.startswith("OPTIMAL_THRESHOLD ="):
            updated_lines.append(f"OPTIMAL_THRESHOLD = {new_threshold:.4f}")
            found = True
        else:
            updated_lines.append(line)
    if not found:
        updated_lines.append(f"OPTIMAL_THRESHOLD = {new_threshold:.4f}")

    with open(config_file, "w") as f:
        f.write("\n".join(updated_lines) + "\n")


def train_and_compare() -> Dict[str, Any]:
    """Execute model training, cross-validation, hyperparameter tuning, and threshold optimization."""
    print("Loading data...")
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        raw_df = load_data()
        df = clean_data(raw_df)

    X_train, X_test, y_train, y_test = prepare_data_split(df)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Calculate scale_pos_weight for XGBoost: negatives / positives
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count
    print(f"Train negatives: {neg_count}, positives: {pos_count}, scale_pos_weight: {scale_pos_weight:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Base pipelines
    prep_pipeline, _, _ = build_full_preprocessing_pipeline()

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", prep_pipeline),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", prep_pipeline),
                ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("prep", prep_pipeline),
                ("clf", XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
    }

    # Evaluate base models
    comparison_records = []
    print("\n--- Evaluating Baseline Models (5-Fold Stratified CV) ---")
    for name, pipe in models.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "f1"],
            n_jobs=-1,
        )
        mean_auc = scores["test_roc_auc"].mean()
        std_auc = scores["test_roc_auc"].std()
        mean_f1 = scores["test_f1"].mean()
        std_f1 = scores["test_f1"].std()
        print(f"{name:18} | ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f}) | F1: {mean_f1:.4f} (+/- {std_f1:.4f})")
        comparison_records.append({
            "model": name,
            "stage": "baseline",
            "mean_roc_auc": round(mean_auc, 4),
            "std_roc_auc": round(std_auc, 4),
            "mean_f1": round(mean_f1, 4),
            "std_f1": round(std_f1, 4),
        })

    # Pick top 2 models by mean_roc_auc
    sorted_models = sorted(comparison_records, key=lambda x: x["mean_roc_auc"], reverse=True)
    top_2_names = [m["model"] for m in sorted_models[:2]]
    print(f"\nTop 2 models selected for tuning: {top_2_names}")

    # Define hyperparameter spaces for top 2
    param_grids = {
        "LogisticRegression": {
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        "RandomForest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [5, 8, 12, 15, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "clf__n_estimators": [100, 150, 200, 250],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
    }

    tuned_results = {}
    for name in top_2_names:
        print(f"\nTuning {name} with RandomizedSearchCV (n_iter=25)...")
        grid = param_grids[name]
        search = RandomizedSearchCV(
            estimator=models[name],
            param_distributions=grid,
            n_iter=25,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_auc = search.best_score_
        best_estimator = search.best_estimator_

        # Calculate CV F1 with default threshold
        oof_proba = cross_val_predict(
            best_estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_f1 = f1_score(y_train, (oof_proba >= 0.5).astype(int))

        print(f"{name} Tuned Best ROC-AUC: {best_auc:.4f} | Default F1: {oof_f1:.4f}")
        print(f"Best params: {search.best_params_}")

        comparison_records.append({
            "model": f"{name} (Tuned)",
            "stage": "tuned",
            "mean_roc_auc": round(best_auc, 4),
            "std_roc_auc": round(search.cv_results_["std_test_score"][search.best_index_], 4),
            "mean_f1": round(oof_f1, 4),
            "std_f1": 0.0,
        })

        tuned_results[name] = {
            "estimator": best_estimator,
            "best_params": search.best_params_,
            "best_auc": best_auc,
            "oof_proba": oof_proba,
        }

    # Pick the best overall model by ROC-AUC
    best_model_name = max(tuned_results.keys(), key=lambda k: tuned_results[k]["best_auc"])
    best_tuned = tuned_results[best_model_name]
    final_pipeline = best_tuned["estimator"]
    best_params = best_tuned["best_params"]
    print(f"\n==========================================")
    print(f"Best overall model: {best_model_name} (ROC-AUC: {best_tuned['best_auc']:.4f})")
    print(f"==========================================")

    # Threshold tuning on out-of-fold validation probabilities
    best_thresh, best_f1, prec_at_best, rec_at_best = tune_decision_threshold(
        y_train.values, best_tuned["oof_proba"]
    )
    print(f"Tuned Optimal Threshold: {best_thresh:.4f}")
    print(f"Validation F1 at optimal threshold: {best_f1:.4f} (Precision: {prec_at_best:.4f}, Recall: {rec_at_best:.4f})")

    # Update threshold in config
    update_config_threshold(best_thresh)

    # Fit final pipeline on all X_train
    print("\nFitting final model on full X_train...")
    final_pipeline.fit(X_train, y_train)

    # Save model pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Model saved to {MODEL_PATH} ({file_size_mb:.2f} MB)")

    # Save comparison table
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df = pd.DataFrame(comparison_records)
    comparison_csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))

    # Save model_card.json
    model_card = {
        "model_name": best_model_name,
        "pipeline_stages": [name for name, _ in final_pipeline.steps],
        "hyperparameters": {k: str(v) for k, v in best_params.items()},
        "optimal_threshold": round(best_thresh, 4),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": XGBClassifier().__module__,
            "joblib": joblib.__version__,
        },
        "validation_metrics": {
            "cv_mean_roc_auc": round(best_tuned["best_auc"], 4),
            "cv_f1_optimal_threshold": round(best_f1, 4),
            "precision_at_optimal_threshold": round(prec_at_best, 4),
            "recall_at_optimal_threshold": round(rec_at_best, 4),
        },
    }
    with open(MODEL_CARD_PATH, "w") as f:
        json.dump(model_card, f, indent=4)
    print(f"Model card saved to {MODEL_CARD_PATH}")

    return {
        "best_model_name": best_model_name,
        "best_threshold": best_thresh,
        "comparison_table": comparison_df,
    }


if __name__ == "__main__":
    train_and_compare()
