"""
Data quality control script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-18
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.eda.data_inspection import load_data, transform_data
from src.utils.data_utils import (
    analyze_target_variable,
    analyze_variable_ranges,
    get_variable_types,
    handle_continuous_outliers,
    handle_missing_target,
)


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

    # # 2. Handle continuous features
    # df_clean, numeric_stats = impute_continuous_features(
    #     df_clean, var_types["continuous"], excluded_cols, target_col
    # )
    # cleaning_stats["numeric_cleaning"] = numeric_stats

    # # 3. Handle categorical features
    # df_clean, categorical_stats = impute_categorical_features(
    #     df_clean, var_types["categorical"], excluded_cols, target_col
    # )
    # cleaning_stats["categorical_cleaning"] = categorical_stats

    # 4. Handle continuous outliers
    _, outlier_stats = handle_continuous_outliers(
        df_clean,
        var_types["continuous"],
        excluded_cols=excluded_cols,
        target_col=target_col,
    )
    cleaning_stats["outlier_analysis"] = outlier_stats

    # 5. Get variable ranges and statistics
    continuous_stats = analyze_variable_ranges(
        df_clean, var_types["continuous"], excluded_cols
    )
    categorical_stats = analyze_variable_ranges(
        df_clean, var_types["categorical"], excluded_cols
    )

    # Extract ranges for printing
    continuous_ranges = {
        var: (stats["min"], stats["max"])
        for var, stats in continuous_stats.items()
        if "min" in stats and "max" in stats
    }
    categorical_levels = {
        var: stats["unique_count"] for var, stats in categorical_stats.items()
    }

    cleaning_stats.update(
        {
            "continuous_stats": continuous_stats,
            "categorical_stats": categorical_stats,
            "continuous_ranges": continuous_ranges,  # Add this for backwards compatibility
            "categorical_levels": categorical_levels,  # Add this for backwards compatibility
            "final_shape": df_clean.shape,
        }
    )

    return df_clean, cleaning_stats


def main():
    """Main function to run data quality checks and cleaning"""

    # Load data using data_inspection functionality
    print("Loading data...")
    df, data_dict = load_data()

    # Transform data using transform_data functionality
    print("Transforming data...")
    df_transformed = transform_data(df)

    # Define columns to exclude and target column
    excluded_cols = ["desc", "member_id", "id"]
    target_col = "bad_flag"

    # Get initial target variable analysis
    target_stats = analyze_target_variable(df_transformed, target_col)

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
