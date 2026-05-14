"""Reusable plotting functions for sleep research analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import find_first_matching_column


def setup_style() -> None:
    """Apply a clean chart style for the notebook."""
    sns.set_theme(style="whitegrid", palette="Set2")


def save_current_plot(filename: str, output_dir: str | Path = "outputs/graphs") -> None:
    """Save the current matplotlib figure to outputs/graphs/."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path / filename, dpi=300, bbox_inches="tight")


def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    filename: str | None = None,
) -> None:
    """Plot a histogram for a numeric column."""
    if column not in df.columns:
        print(f"Column not found for distribution plot: {column}")
        return

    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Column is not numeric, skipping histogram: {column}")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")

    if filename:
        save_current_plot(filename)

    plt.show()


def plot_categorical_count(
    df: pd.DataFrame,
    column: str,
    filename: str | None = None,
) -> None:
    """Plot counts for a categorical or discrete column."""
    if column not in df.columns:
        print(f"Column not found for count plot: {column}")
        return

    plt.figure(figsize=(9, 5))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, order=order)
    plt.title(f"Count of {column.replace('_', ' ').title()}")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")

    if filename:
        save_current_plot(filename)

    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    filename: str | None = "correlation_heatmap.png",
) -> None:
    """Plot a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        print("Need at least two numeric columns to create a correlation heatmap.")
        return

    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")

    if filename:
        save_current_plot(filename)

    plt.show()


def plot_research_variables(df: pd.DataFrame) -> None:
    """Create plots for common smart alarm research variables if available."""
    plot_specs = [
        ("sleep_duration", ["sleep_duration", "duration"], "sleep_duration_distribution.png"),
        ("heart_rate", ["heart_rate", "heart", "pulse"], "heart_rate_distribution.png"),
        ("stress_level", ["stress_level", "stress"], "stress_level_distribution.png"),
        (
            "physical_activity",
            ["physical_activity", "activity", "steps"],
            "physical_activity_distribution.png",
        ),
        ("sleep_quality", ["sleep_quality", "quality_of_sleep", "quality"], "sleep_quality_count.png"),
    ]

    for label, keywords, filename in plot_specs:
        column = find_first_matching_column(df, keywords)

        if column is None:
            print(f"No column found for {label}; skipping this plot.")
            continue

        if pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique() > 10:
            plot_numeric_distribution(df, column, filename)
        else:
            plot_categorical_count(df, column, filename)
