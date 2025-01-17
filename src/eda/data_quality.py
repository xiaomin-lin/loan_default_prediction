"""
Data quality control script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-18
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from data_inspection import (
    analyze_target_variable,
    get_variable_types,
    load_data,
    transform_data,
)


def handle_missing_target(
    df: pd.DataFrame,
    target_col: str = "bad_flag",
    target_type: type = int,
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove rows with missing target variable and convert to correct type
    Args:
        df: Input DataFrame
        target_col: Name of target column (default: "bad_flag")
        target_type: Type to convert target to (default: int for classification)
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

    # Make a deep copy and handle excluded columns
    processed_df = df.copy(deep=True)
    if excluded_cols:
        processed_df = processed_df.drop(columns=excluded_cols, errors="ignore")

    stats = {
        "original_rows": len(processed_df),
        "missing_target": processed_df[target_col].isnull().sum(),
        "original_dtypes": str(processed_df[target_col].dtype),
    }

    # Remove rows with missing target
    processed_df = processed_df.dropna(subset=[target_col])

    # Convert target to specified type
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


def impute_continuous_features(
    df: pd.DataFrame,
    var_types: Dict,
    excluded_cols: Optional[List[str]] = None,
    method: str = "median",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for continuous features
    Args:
        df: Input DataFrame
        var_types: Dictionary containing variable types from get_variable_types()
        excluded_cols: List of columns to exclude from imputation (default: None)
        method: Imputation method ('median' or 'mean', default: 'median')
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)
    stats = {"imputed_columns": {}}

    if excluded_cols:
        continuous_cols = [
            col
            for col in var_types["continuous"]
            if col not in excluded_cols and col != "bad_flag"
        ]
    else:
        continuous_cols = [col for col in var_types["continuous"] if col != "bad_flag"]

    for col in continuous_cols:
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
    var_types: Dict,
    excluded_cols: Optional[List[str]] = None,
    method: str = "mode",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for categorical features
    Args:
        df: Input DataFrame
        var_types: Dictionary containing variable types from get_variable_types()
        excluded_cols: List of columns to exclude from imputation (default: None)
        method: Imputation method (currently only 'mode' supported)
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)
    stats = {"imputed_columns": {}}

    if excluded_cols:
        categorical_cols = [
            col for col in var_types["categorical"] if col not in excluded_cols
        ]
    else:
        categorical_cols = var_types["categorical"]

    for col in categorical_cols:
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
    var_types: Dict,
    threshold: float = 1.5,
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify outliers in continuous features based on the IQR method.
    Args:
        df: Input DataFrame
        var_types: Dictionary containing variable types from get_variable_types()
        threshold: IQR multiplier for outlier detection (default: 1.5)
        excluded_cols: List of columns to exclude from outlier detection (default: None)
    Returns:
        tuple: (processed_df, stats)
    """
    stats = {"outliers": {}}
    processed_df = df.copy()

    if excluded_cols:
        continuous_cols = [
            col
            for col in var_types["continuous"]
            if col not in excluded_cols and col != "bad_flag"
        ]
    else:
        continuous_cols = [col for col in var_types["continuous"] if col != "bad_flag"]

    for col in continuous_cols:
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


def get_continuous_variable_ranges(
    df: pd.DataFrame, continuous_vars: list, excluded_cols: Optional[List[str]] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Get the value range (min and max) for each continuous variable.
    Args:
        df: The input DataFrame
        continuous_vars: List of continuous variable column names
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        Dict[str, Tuple[float, float]]: Dictionary with variable names as keys and (min, max) tuples as values
    """
    if excluded_cols:
        continuous_vars = [col for col in continuous_vars if col not in excluded_cols]

    ranges = {}
    for var in continuous_vars:
        non_null_values = df[var].dropna()
        min_val = non_null_values.min()
        max_val = non_null_values.max()
        total_count = len(df[var])
        non_null_count = len(non_null_values)
        missing_count = total_count - non_null_count

        ranges[var] = (min_val, max_val)
        print(
            f"Variable '{var}': Min = {min_val}, Max = {max_val} "
            f"(based on {non_null_count:,} non-null values, {missing_count:,} missing)"
        )
    return ranges


def get_categorical_variable_levels(
    df: pd.DataFrame, categorical_vars: list, excluded_cols: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Get the number of unique values (levels) for each categorical variable.
    Args:
        df: The input DataFrame
        categorical_vars: List of categorical variable column names
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        Dict[str, int]: Dictionary with variable names as keys and their number of unique levels as values
    """
    if excluded_cols:
        categorical_vars = [col for col in categorical_vars if col not in excluded_cols]

    levels = {}
    for var in categorical_vars:
        non_null_values = df[var].dropna()
        unique_count = non_null_values.nunique()
        total_count = len(df[var])
        non_null_count = len(non_null_values)
        missing_count = total_count - non_null_count

        levels[var] = unique_count
        print(
            f"Variable '{var}': {unique_count} unique levels "
            f"(based on {non_null_count:,} non-null values, {missing_count:,} missing)"
        )
    return levels


def clean_dataset(
    df: pd.DataFrame,
    target_col: str = "bad_flag",
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to clean the dataset
    Args:
        df: Input DataFrame
        target_col: Name of target column (default: "bad_flag")
        excluded_cols: List of columns to exclude from cleaning (default: None)
    Returns:
        tuple: (cleaned_df, cleaning_stats)
    """
    cleaning_stats = {"original_shape": df.shape}

    # Get variable types
    var_types = get_variable_types(df, excluded_cols)

    # 1. Handle missing target
    df_clean, target_stats = handle_missing_target(
        df=df, target_col=target_col, target_type=int, excluded_cols=excluded_cols
    )
    cleaning_stats["target_cleaning"] = target_stats

    # 2. Handle continuous features
    # df_clean, numeric_stats = impute_continuous_features(
    #     df_clean, var_types, excluded_cols
    # )
    # cleaning_stats["numeric_cleaning"] = numeric_stats

    # 3. Handle categorical features
    # df_clean, categorical_stats = impute_categorical_features(
    #     df_clean, var_types, excluded_cols
    # )
    # cleaning_stats["categorical_cleaning"] = categorical_stats

    # 4. Handle continuous outliers
    _, outlier_stats = handle_continuous_outliers(
        df_clean, var_types, excluded_cols=excluded_cols
    )
    cleaning_stats["outlier_analysis"] = outlier_stats

    # 5. Get continuous variable ranges
    continuous_ranges = get_continuous_variable_ranges(
        df_clean, var_types["continuous"], excluded_cols
    )
    cleaning_stats["continuous_ranges"] = continuous_ranges

    # 6. Get categorical variable levels
    categorical_levels = get_categorical_variable_levels(
        df_clean, var_types["categorical"], excluded_cols
    )
    cleaning_stats["categorical_levels"] = categorical_levels

    cleaning_stats["final_shape"] = df_clean.shape

    return df_clean, cleaning_stats


def main():
    """Main function to run data quality checks and cleaning"""
    # Define columns to exclude and target column
    excluded_cols = ["desc", "member_id", "id"]
    target_col = "bad_flag"

    # Load data using data_inspection functionality
    print("Loading data...")
    df, data_dict = load_data(excluded_cols=excluded_cols)

    # Transform data using transform_data functionality
    print("Transforming data...")
    df_transformed = transform_data(df, excluded_cols=excluded_cols)

    # Get initial target variable analysis
    target_stats = analyze_target_variable(df_transformed, target_col=target_col)
    print("\n=== Initial Target Variable Analysis ===")
    print("Class distribution before cleaning:")
    for label, count in target_stats["distribution"].items():
        percentage = target_stats["distribution_percentage"][label] * 100
        print(f"Class {label}: {count} ({percentage:.2f}%)")

    # Clean dataset
    print("\n=== Cleaning Dataset ===")
    cleaned_df, cleaning_stats = clean_dataset(
        df_transformed, target_col=target_col, excluded_cols=excluded_cols
    )

    # Print cleaning summary
    print("\n=== Cleaning Summary ===")
    print(f"Original shape: {cleaning_stats['original_shape']}")
    print(f"Final shape: {cleaning_stats['final_shape']}")

    # Target variable cleaning
    target_stats = cleaning_stats["target_cleaning"]
    print(
        f"\nRows with missing target removed: {target_stats['removed_rows']} "
        f"({target_stats['removed_percentage']:.2f}%)"
    )

    # Outlier summary
    print("\nOutlier Analysis:")
    for col, stats in cleaning_stats["outlier_analysis"]["outliers"].items():
        if stats["count"] > 0:
            print(f"- {col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")

    # Continuous variable ranges
    print("\nContinuous Variable Ranges:")
    for var, (min_val, max_val) in cleaning_stats["continuous_ranges"].items():
        print(f"- {var}: Min = {min_val}, Max = {max_val}")

    # Categorical variable levels
    print("\nCategorical Variable Levels:")
    for var, unique_count in cleaning_stats["categorical_levels"].items():
        print(f"- {var}: {unique_count} unique levels")


if __name__ == "__main__":
    main()
