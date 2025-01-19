"""
Data quality control script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-18
"""

from src.eda.data_inspection import load_data, transform_data
from src.utils.data_utils import analyze_target_variable, clean_data


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
    cleaned_df, cleaning_stats = clean_data(
        df=df_transformed,
        id_col="member_id",
        target_col=target_col,
        excluded_cols=excluded_cols,
        dedup_strategy="first",
        verbose=True,
    )


if __name__ == "__main__":
    main()
