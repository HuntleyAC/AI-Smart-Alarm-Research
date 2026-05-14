"""Baseline machine learning models for the smart alarm research project."""

from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(task_type: str) -> dict[str, object]:
    """Create baseline models for classification or regression."""
    if task_type == "classification":
        return {
            "logistic_regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
        }

    return {
        "linear_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
    }


def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Train baseline models and return fitted models plus metrics."""
    if X is None or y is None or X.empty:
        print("Model training skipped because features or target are missing.")
        return {}, {}

    stratify = y if task_type == "classification" and y.nunique() > 1 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        print("Stratified split failed, using a regular train/test split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    models = build_models(task_type)
    fitted_models = {}
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        fitted_models[model_name] = model

        if task_type == "classification":
            results[model_name] = {
                "accuracy": accuracy_score(y_test, predictions),
                "classification_report": classification_report(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }
        else:
            results[model_name] = {
                "mae": mean_absolute_error(y_test, predictions),
                "mse": mean_squared_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
            }

    return fitted_models, results


def save_model(
    model: object,
    filename: str,
    output_dir: str | Path = "outputs/models",
) -> Path:
    """Save a trained model as a pickle file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / filename

    with model_path.open("wb") as file:
        pickle.dump(model, file)

    print(f"Model saved to {model_path}")
    return model_path


def choose_best_model(
    models: dict[str, object],
    results: dict[str, dict[str, object]],
    task_type: str,
) -> tuple[str | None, object | None]:
    """Choose the best baseline model using accuracy or R2."""
    if not models or not results:
        print("No trained models available.")
        return None, None

    metric = "accuracy" if task_type == "classification" else "r2"
    best_name = max(results, key=lambda name: results[name][metric])
    return best_name, models[best_name]
