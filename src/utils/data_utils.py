from typing import Dict, List, Optional

import pandas as pd


def get_duplicates_by_column(
    df: pd.DataFrame, column: str, verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Check for duplicates in a DataFrame based on a specific column and return the duplicate entries
    with all their columns for comparison.

    Args:
        df: Input pandas DataFrame
        column: Column name to check for duplicates
        verbose: Whether to print duplicate information (default: True)

    Returns:
        Optional[pd.DataFrame]: DataFrame containing all rows with duplicate values in the specified column,
                              or None if no duplicates found
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    duplicate_mask = df.duplicated(subset=column, keep=False)
    duplicates_df = df[duplicate_mask].sort_values(by=column)

    if duplicates_df.empty:
        if verbose:
            print(f"\nNo duplications found based on column '{column}'")
        return None

    if verbose:
        print(f"\n=== Duplicate Analysis for column '{column}' ===")
        print(f"Number of duplicate values: {duplicates_df[column].nunique()}")
        print(f"Total rows with duplicates: {len(duplicates_df)}")

    return duplicates_df


def get_basic_info(df: pd.DataFrame, excluded_cols: Optional[List[str]] = None) -> Dict:
    """
    Get basic information about the dataset
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        dict: Basic dataset information including shape, columns, dtypes, and missing values
    """
    if excluded_cols:
        df = df.drop(columns=excluded_cols, errors="ignore")

    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    return info


def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Analyze the target variable distribution and missing values
    Args:
        df: pandas DataFrame
        target_col: Name of target column
    Returns:
        dict: Target variable statistics including distribution and missing values
    """
    target_stats = {
        "distribution": df[target_col].value_counts().to_dict(),
        "distribution_percentage": df[target_col]
        .value_counts(normalize=True)
        .to_dict(),
        "null_count": df[target_col].isnull().sum(),
    }
    return target_stats


def get_variable_types(
    df: pd.DataFrame, excluded_cols: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Categorize variables by their types (continuous vs categorical)
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        dict: Variables categorized by type
    """
    if excluded_cols is None:
        excluded_cols = []

    df_filtered = df.drop(columns=excluded_cols, errors="ignore")

    continuous_cols = df_filtered.select_dtypes(
        include=["Int64", "Float64", "Int32", "Float32"]
    ).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(
        include=["object", "category", "string", "bool", "boolean"]
    ).columns.tolist()

    return {
        "continuous": continuous_cols,
        "categorical": categorical_cols,
    }
