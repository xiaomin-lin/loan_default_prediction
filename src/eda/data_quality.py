"""
Data quality control script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-18
"""

from typing import Dict, Tuple

import pandas as pd
from data_inspection import analyze_target_variable, get_variable_types, load_data


def handle_missing_target(
    df: pd.DataFrame, target_col: str, target_type: type = int
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove rows with missing target variable and convert to correct type
    Args:
        df: Input DataFrame
        target_col: Name of target column
        target_type: Type to convert target to (default: int for classification)
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

    stats = {
        "original_rows": len(df),
        "missing_target": df[target_col].isnull().sum(),
        "original_dtypes": str(df[target_col].dtype),
    }

    # Make a deep copy and remove rows with missing target
    processed_df = df.copy(deep=True)
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
    df: pd.DataFrame, var_types: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for continuous features
    Args:
        df: Input DataFrame
        var_types: Dictionary containing variable types from get_variable_types()
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)  # Make a deep copy
    stats = {"imputed_columns": {}}

    for col in var_types["continuous"]:
        if col != "bad_flag":  # Skip target variable
            missing_count = processed_df[col].isnull().sum()
            if missing_count > 0:
                median_value = processed_df[col].median()
                processed_df.loc[processed_df[col].isnull(), col] = median_value
                stats["imputed_columns"][col] = {
                    "missing_count": missing_count,
                    "imputed_value": median_value,
                    "imputation_method": "median",
                }

    return processed_df, stats


def impute_categorical_features(
    df: pd.DataFrame, var_types: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for categorical features
    Args:
        df: Input DataFrame
        var_types: Dictionary containing variable types from get_variable_types()
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)  # Make a deep copy
    stats = {"imputed_columns": {}}

    for col in var_types["categorical"]:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            mode_value = processed_df[col].mode()[0]
            processed_df.loc[processed_df[col].isnull(), col] = mode_value
            stats["imputed_columns"][col] = {
                "missing_count": missing_count,
                "imputed_value": mode_value,
                "imputation_method": "mode",
            }

    return processed_df, stats


def handle_outliers(
    df: pd.DataFrame, var_types: Dict, threshold: float = 1.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify outliers in continuous features
    Args:
        df: Input DataFrame
        var_types: Dictionary containing variable types from get_variable_types()
        threshold: IQR multiplier for outlier detection
    Returns:
        tuple: (processed_df, stats)
    """
    stats = {"outliers": {}}
    processed_df = df.copy()

    for col in var_types["continuous"]:
        if col != "bad_flag":  # Skip target variable
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


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to clean the dataset
    Args:
        df: Input DataFrame
    Returns:
        tuple: (cleaned_df, cleaning_stats)
    """
    cleaning_stats = {"original_shape": df.shape}

    # Get variable types
    var_types = get_variable_types(df)

    # 1. Handle missing target
    df_clean, target_stats = handle_missing_target(
        df=df, target_col="bad_flag", target_type=int  # For binary classification
    )
    cleaning_stats["target_cleaning"] = target_stats

    # 2. Handle continuous features
    # df_clean, numeric_stats = impute_continuous_features(df_clean, var_types)
    # cleaning_stats["numeric_cleaning"] = numeric_stats

    # 3. Handle categorical features
    # df_clean, categorical_stats = impute_categorical_features(df_clean, var_types)
    # cleaning_stats["categorical_cleaning"] = categorical_stats

    # 4. Handle outliers (identification only)
    _, outlier_stats = handle_outliers(df_clean, var_types)
    cleaning_stats["outlier_analysis"] = outlier_stats

    cleaning_stats["final_shape"] = df_clean.shape

    return df_clean, cleaning_stats


def main():
    """Main function to run data quality checks and cleaning"""
    # Load data using data_inspection functionality
    print("Loading data...")
    df, data_dict = load_data()

    # Get initial target variable analysis
    target_stats = analyze_target_variable(df)
    print("\n=== Initial Target Variable Analysis ===")
    print("Class distribution before cleaning:")
    for label, count in target_stats["distribution"].items():
        percentage = target_stats["distribution_percentage"][label] * 100
        print(f"Class {label}: {count} ({percentage:.2f}%)")

    # Clean dataset
    print("\n=== Cleaning Dataset ===")
    cleaned_df, cleaning_stats = clean_dataset(df)

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

    # Feature cleaning
    print("\nImputed continuous features:")
    for col, stats in (
        cleaning_stats.get("numeric_cleaning", {}).get("imputed_columns", {}).items()
    ):
        print(f"- {col}: {stats['missing_count']} values imputed with median")

    print("\nImputed categorical features:")
    for col, stats in (
        cleaning_stats.get("categorical_cleaning", {})
        .get("imputed_columns", {})
        .items()
    ):
        print(f"- {col}: {stats['missing_count']} values imputed with mode")

    # Outlier summary
    print("\nOutlier Analysis:")
    for col, stats in cleaning_stats["outlier_analysis"]["outliers"].items():
        if stats["count"] > 0:
            print(f"- {col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")


if __name__ == "__main__":
    main()
