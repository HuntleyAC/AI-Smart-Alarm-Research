"""Preprocessing helpers for sleep, lifestyle, and wearable datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the DataFrame with simple snake_case column names."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using beginner-friendly defaults.

    Numeric columns are filled with the median. Categorical columns are filled
    with the most frequent value. If a categorical column has no mode, it gets
    the value "unknown".
    """
    cleaned = df.copy()

    for column in cleaned.columns:
        if cleaned[column].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        else:
            mode_values = cleaned[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
            cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned


def encode_categorical_variables(
    df: pd.DataFrame,
    exclude_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Encode text columns with LabelEncoder.

    Parameters
    ----------
    df:
        Cleaned DataFrame.
    exclude_columns:
        Columns that should not be encoded, such as the target column.

    Returns
    -------
    tuple
        Encoded DataFrame and a dictionary of fitted encoders.
    """
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    excluded = set(exclude_columns or [])

    for column in encoded.select_dtypes(include=["object", "category", "bool"]).columns:
        if column in excluded:
            continue

        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column].astype(str))
        encoders[column] = encoder

    return encoded, encoders


def find_first_matching_column(
    df: pd.DataFrame,
    keywords: Iterable[str],
) -> str | None:
    """Find the first column whose name contains one of the given keywords."""
    normalized_keywords = [keyword.lower().replace(" ", "_") for keyword in keywords]

    for column in df.columns:
        normalized_column = column.lower().replace(" ", "_")
        if any(keyword in normalized_column for keyword in normalized_keywords):
            return column

    return None


def choose_target_column(df: pd.DataFrame) -> str | None:
    """Choose a likely ML target from common sleep datasets.

    Priority:
    1. sleep_quality
    2. sleep_disorder
    3. sleep_efficiency
    """
    target_options = [
        ["sleep_quality", "quality_of_sleep", "quality"],
        ["sleep_disorder", "disorder"],
        ["sleep_efficiency", "efficiency"],
    ]

    for keywords in target_options:
        target = find_first_matching_column(df, keywords)
        if target is not None:
            return target

    print(
        "No target column found. Expected something like Sleep Quality, "
        "Sleep Disorder, or Sleep Efficiency."
    )
    return None


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None, str | None, str | None]:
    """Prepare X and y for machine learning.

    Returns X, y, selected target column, and task type ("classification" or
    "regression"). If preparation is not possible, returns None values with a
    clear message.
    """
    if df.empty:
        print("The DataFrame is empty. Load a dataset before modeling.")
        return None, None, None, None

    cleaned = handle_missing_values(clean_column_names(df))
    target = target_column or choose_target_column(cleaned)

    if target is None or target not in cleaned.columns:
        print("Cannot prepare model data because no valid target column was found.")
        return None, None, None, None

    y = cleaned[target]
    X = cleaned.drop(columns=[target])

    # Remove ID-like columns that usually do not help prediction.
    id_columns = [column for column in X.columns if column == "id" or column.endswith("_id")]
    if id_columns:
        X = X.drop(columns=id_columns)

    X, _ = encode_categorical_variables(X)

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
        task_type = "regression"
    else:
        task_type = "classification"
        if not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y.astype(str)), name=target)

    return X, y, target, task_type
