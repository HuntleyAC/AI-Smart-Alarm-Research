"""Utilities for loading sleep and wearable CSV datasets.

The project is designed so you can replace the CSV file later without
rewriting the notebook. Keep raw data in data/raw/.
"""

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file safely and return a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Relative or absolute path to a CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset. If the file cannot be loaded, an empty DataFrame is
        returned and a clear message is printed.
    """
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {path}")
        print("Place your CSV file in data/raw/ and update DATA_PATH in the notebook.")
        return pd.DataFrame()

    if path.suffix.lower() != ".csv":
        print(f"Expected a .csv file, but received: {path.name}")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"The file is empty: {path}")
    except pd.errors.ParserError:
        print(f"Could not parse the CSV file: {path}")
    except UnicodeDecodeError:
        print(f"Encoding error while reading: {path}")
        print("Try saving the file as UTF-8 CSV.")
    except Exception as error:
        print(f"Unexpected error while loading {path}: {error}")

    return pd.DataFrame()


def list_csv_files(folder_path: str | Path = "data/raw") -> list[Path]:
    """List CSV files in a folder so the user can choose a dataset."""
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return []

    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {folder}.")

    return csv_files
