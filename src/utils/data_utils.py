from typing import Dict, List, Optional, Tuple

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


def impute_continuous_features(
    df: pd.DataFrame,
    continuous_cols: List[str],
    excluded_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    method: str = "median",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for continuous features
    Args:
        df: Input DataFrame
        continuous_cols: List of continuous column names
        excluded_cols: List of columns to exclude from imputation (default: None)
        target_col: Name of target column to exclude (default: None)
        method: Imputation method ('median' or 'mean', default: 'median')
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)
    stats = {"imputed_columns": {}}

    cols_to_impute = continuous_cols
    if excluded_cols:
        cols_to_impute = [col for col in cols_to_impute if col not in excluded_cols]
    if target_col:
        cols_to_impute = [col for col in cols_to_impute if col != target_col]

    for col in cols_to_impute:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            if method == "mean":
                imputed_value = processed_df[col].mean()
            else:  # default to median
                imputed_value = processed_df[col].median()

            processed_df.loc[processed_df[col].isnull(), col] = imputed_value
            stats["imputed_columns"][col] = {
                "missing_count": missing_count,
                "imputed_value": imputed_value,
                "imputation_method": method,
            }

    return processed_df, stats


def impute_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    excluded_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    method: str = "mode",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for categorical features
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        excluded_cols: List of columns to exclude from imputation (default: None)
        target_col: Name of target column to exclude (default: None)
        method: Imputation method (currently only 'mode' supported)
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)
    stats = {"imputed_columns": {}}

    cols_to_impute = categorical_cols
    if excluded_cols:
        cols_to_impute = [col for col in cols_to_impute if col not in excluded_cols]
    if target_col:
        cols_to_impute = [col for col in cols_to_impute if col != target_col]

    for col in cols_to_impute:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            mode_value = processed_df[col].mode()[0]
            processed_df.loc[processed_df[col].isnull(), col] = mode_value
            stats["imputed_columns"][col] = {
                "missing_count": missing_count,
                "imputed_value": mode_value,
                "imputation_method": method,
            }

    return processed_df, stats


def handle_continuous_outliers(
    df: pd.DataFrame,
    continuous_cols: List[str],
    threshold: float = 1.5,
    excluded_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify outliers in continuous features based on the IQR method.
    Args:
        df: Input DataFrame
        continuous_cols: List of continuous column names
        threshold: IQR multiplier for outlier detection (default: 1.5)
        excluded_cols: List of columns to exclude from outlier detection (default: None)
        target_col: Name of target column to exclude (default: None)
    Returns:
        tuple: (processed_df, stats)
    """
    stats = {"outliers": {}}
    processed_df = df.copy()

    cols_to_analyze = continuous_cols
    if excluded_cols:
        cols_to_analyze = [col for col in cols_to_analyze if col not in excluded_cols]
    if target_col:
        cols_to_analyze = [col for col in cols_to_analyze if col != target_col]

    for col in cols_to_analyze:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        stats["outliers"][col] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(df)) * 100,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    return processed_df, stats


def analyze_variable_ranges(
    df: pd.DataFrame,
    columns: List[str],
    excluded_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Get the value range and statistics for specified variables.
    Args:
        df: The input DataFrame
        columns: List of column names to analyze
        excluded_cols: List of columns to exclude from analysis (default: None)
        verbose: Whether to print analysis information (default: True)
    Returns:
        Dict[str, Dict]: Dictionary with variable statistics
    """
    if excluded_cols:
        columns = [col for col in columns if col not in excluded_cols]

    stats = {}
    for col in columns:
        non_null_values = df[col].dropna()
        total_count = len(df[col])
        non_null_count = len(non_null_values)
        missing_count = total_count - non_null_count

        col_stats = {
            "unique_count": non_null_values.nunique(),
            "total_count": total_count,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_percentage": (missing_count / total_count) * 100,
        }

        # Add min/max for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "min": non_null_values.min(),
                    "max": non_null_values.max(),
                }
            )

        stats[col] = col_stats

        if verbose:
            if pd.api.types.is_numeric_dtype(df[col]):
                print(
                    f"Variable '{col}': Min = {col_stats['min']}, Max = {col_stats['max']} "
                    f"(based on {non_null_count:,} non-null values, {missing_count:,} missing)"
                )
            else:
                print(
                    f"Variable '{col}': {col_stats['unique_count']} unique levels "
                    f"(based on {non_null_count:,} non-null values, {missing_count:,} missing)"
                )

    return stats


def handle_missing_target(
    df: pd.DataFrame,
    target_col: str,
    target_type: type = int,
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove rows with missing target variable and convert to correct type
    Args:
        df: Input DataFrame
        target_col: Name of target column
        target_type: Type to convert target to (default: int)
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        tuple: (processed_df, stats) where stats contains information about the cleaning process
    Raises:
        ValueError: If target_col is not in DataFrame
        TypeError: If target_type is not supported
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    if target_type not in [int, float, str, bool]:
        raise TypeError(f"Unsupported target type: {target_type}")

    processed_df = df.copy(deep=True)
    if excluded_cols:
        processed_df = processed_df.drop(columns=excluded_cols, errors="ignore")

    stats = {
        "original_rows": len(processed_df),
        "missing_target": processed_df[target_col].isnull().sum(),
        "original_dtypes": str(processed_df[target_col].dtype),
    }

    processed_df = processed_df.dropna(subset=[target_col])

    try:
        processed_df.loc[:, target_col] = processed_df[target_col].astype(target_type)
    except Exception as e:
        raise TypeError(f"Failed to convert target to {target_type}: {str(e)}")

    stats.update(
        {
            "remaining_rows": len(processed_df),
            "removed_rows": stats["original_rows"] - len(processed_df),
            "removed_percentage": (stats["original_rows"] - len(processed_df))
            / stats["original_rows"]
            * 100,
            "final_dtype": str(processed_df[target_col].dtype),
        }
    )

    return processed_df, stats
