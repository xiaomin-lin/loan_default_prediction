"""
Initial data inspection script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-17
"""

import json
from pathlib import Path

import pandas as pd


def load_data():
    """
    Load training data and data dictionary
    Returns:
        tuple: (DataFrame, dict) containing training data and data dictionary
    """
    # Get current working directory
    root_dir = Path.cwd()

    # Load training data, skipping the first row and using the second row as header
    df = pd.read_csv(root_dir / "data/training_loan_data.csv", skiprows=1, header=0)

    # Drop any columns that are completely empty or unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Load data dictionary
    with open(root_dir / "data/dict_data.json", "r") as f:
        data_dict = json.load(f)

    # Verify columns match data dictionary
    dict_cols = set(data_dict["dictionary"].keys())
    df_cols = set(df.columns)

    print("\n=== Column Verification ===")
    print(
        f"Dataset shape: {df.shape}"
    )  # Should show correct number of rows and 23 columns
    print(f"Number of columns in data dictionary: {len(dict_cols)}")
    print(f"Number of columns in dataset: {len(df_cols)}")

    # Print all columns for verification
    print("\nColumns in DataFrame:")
    for col in sorted(df.columns):
        if col in data_dict["dictionary"]:
            print(f"✓ {col}: {data_dict['dictionary'][col]}")
        else:
            print(f"✗ {col}: Not found in dictionary")

    return df, data_dict


def get_basic_info(df):
    """
    Get basic information about the dataset
    Args:
        df: pandas DataFrame
    Returns:
        dict: Basic dataset information
    """
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
    return info


def analyze_target_variable(df):
    """
    Analyze the target variable (bad_flag)
    Args:
        df: pandas DataFrame
    Returns:
        dict: Target variable statistics
    """
    target_stats = {
        "distribution": df["bad_flag"].value_counts().to_dict(),
        "distribution_percentage": df["bad_flag"]
        .value_counts(normalize=True)
        .to_dict(),
        "null_count": df["bad_flag"].isnull().sum(),
    }
    return target_stats


def get_variable_types(df, excluded_cols=None):
    """
    Categorize variables by their types, excluding specified columns
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        dict: Variables categorized by type
    """
    if excluded_cols is None:
        excluded_cols = []

    # Exclude specified columns from the DataFrame
    df_filtered = df.drop(columns=excluded_cols, errors="ignore")

    numeric_cols = df_filtered.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
    }


def check_duplicate_member_ids(df):
    """
    Check for duplicate member_ids in the dataset
    Args:
        df: pandas DataFrame
    Returns:
        None
    """
    duplicate_members = df[df.duplicated(subset="member_id", keep=False)]

    if not duplicate_members.empty:
        print("\n=== Duplicate Member IDs ===")
        print(duplicate_members[["member_id"]].drop_duplicates())
        print(f"Total duplicate member_ids: {duplicate_members['member_id'].nunique()}")
    else:
        print("\nNo duplicate member_ids found.")


def transform_data(df):
    """
    Transform specified columns in the DataFrame to the desired data types.
    Args:
        df: pandas DataFrame
    Returns:
        pandas DataFrame: Transformed DataFrame
    """
    # Transforming the specified columns
    df["id"] = df["id"].astype(str)
    df["member_id"] = df["member_id"].fillna(0).astype(int)
    df["loan_amnt"] = df["loan_amnt"].astype(int)
    df["term"] = df["term"].astype(str)
    # df['int_rate'] = df['int_rate'].astype(float)
    df["emp_length"] = df["emp_length"].astype(str)

    return df


def main():
    """Main function to run initial data inspection"""
    # Load data
    print("Loading data and verifying columns...")
    df, data_dict = load_data()

    # Transform the data
    df = transform_data(df)

    print(df.head())

    # Check for duplicate member_ids
    check_duplicate_member_ids(df)

    # Get basic information
    basic_info = get_basic_info(df)
    print("\n=== Basic Dataset Information ===")
    print(f"Dataset shape: {basic_info['shape']}")

    # Display column descriptions from data dictionary
    print("\n=== Column Descriptions ===")
    for col in df.columns:
        if col in data_dict["dictionary"]:
            print(f"{col}: {data_dict['dictionary'][col]}")

    print("\n=== Missing Values ===")
    for col, count in basic_info["missing_values"].items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"{col}: {count} ({percentage:.2f}%)")

    # Analyze target variable
    target_stats = analyze_target_variable(df)
    print("\n=== Target Variable Analysis ===")
    print("Class distribution:")
    for label, count in target_stats["distribution"].items():
        percentage = target_stats["distribution_percentage"][label] * 100
        print(f"Class {label}: {count} ({percentage:.2f}%)")

    # Get variable types
    var_types = get_variable_types(df)
    print("\n=== Variable Types ===")
    print(
        f"Numeric variables ({len(var_types['numeric'])}):",
        ", ".join(var_types["numeric"]),
    )
    print(
        f"\nCategorical variables ({len(var_types['categorical'])}):",
        ", ".join(var_types["categorical"]),
    )


if __name__ == "__main__":
    main()
