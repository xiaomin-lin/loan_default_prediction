"""
Initial data inspection script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-17
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.data_utils import (
    analyze_target_variable,
    get_basic_info,
    get_variable_types,
)


def load_data(excluded_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load training data and data dictionary
    Args:
        excluded_cols: List of columns to exclude from loading (default: None)
    Returns:
        tuple: (DataFrame, dict) containing training data and data dictionary
    """
    # Get current working directory
    root_dir = Path.cwd()

    # Load training data
    df = pd.read_csv(root_dir / "data/training_loan_data.csv", skiprows=1, header=0)

    # Drop any columns that are completely empty or unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Drop excluded columns if specified
    if excluded_cols:
        df = df.drop(columns=excluded_cols, errors="ignore")

    # Load data dictionary
    with open(root_dir / "data/dict_data.json", "r") as f:
        data_dict = json.load(f)

    # Verify columns match data dictionary
    dict_cols = set(data_dict["dictionary"].keys())
    if excluded_cols:
        dict_cols = dict_cols - set(excluded_cols)
    df_cols = set(df.columns)

    print("\n=== Column Verification ===")
    print(f"Dataset shape: {df.shape}")
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


def transform_data(
    df: pd.DataFrame, excluded_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Transform specified columns in the DataFrame to the desired data types.
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from transformation (default: None)
    Returns:
        pandas DataFrame: Transformed DataFrame
    """
    df_transformed = df.copy()

    if excluded_cols:
        columns_to_transform = set(df.columns) - set(excluded_cols)
    else:
        columns_to_transform = set(df.columns)

    # Transform specific columns
    if "id" in columns_to_transform:
        df_transformed["id"] = df_transformed["id"].astype(str)
    if "member_id" in columns_to_transform:
        df_transformed["member_id"] = df_transformed["member_id"].astype(str)
    if "loan_amnt" in columns_to_transform:
        df_transformed["loan_amnt"] = df_transformed["loan_amnt"].astype(int)
    if "term" in columns_to_transform:
        df_transformed["term"] = df_transformed["term"].str.strip().astype("category")
    if "int_rate" in columns_to_transform:
        df_transformed["int_rate"] = (
            df_transformed["int_rate"].str.replace("%", "").astype(float) / 100.0
        )
    if "revol_util" in columns_to_transform:
        df_transformed["revol_util"] = (
            df_transformed["revol_util"].str.replace("%", "").astype(float) / 100.0
        )
    if "emp_length" in columns_to_transform:
        df_transformed["emp_length"] = df_transformed["emp_length"].apply(
            lambda x: int("".join(filter(str.isdigit, x))) if pd.notnull(x) else None
        )
    if "bad_flag" in columns_to_transform:
        df_transformed["bad_flag"] = (
            df_transformed["bad_flag"].map({1.0: 1, 0.0: 0}).astype("Int64")
        )
        df_transformed["bad_flag"] = df_transformed["bad_flag"].astype("category")

    return df_transformed


def main():
    """Main function to run initial data inspection"""
    print("Loading data and verifying columns...")
    df, data_dict = load_data()

    # Transform the data
    df = transform_data(df)

    # Define columns to exclude
    excluded_cols = ["desc", "member_id", "id"]
    target_col = "bad_flag"

    # Get basic information
    basic_info = get_basic_info(df, excluded_cols=excluded_cols)
    print("\n=== Basic Dataset Information ===")
    print(f"Dataset shape: {basic_info['shape']}")

    # Display column descriptions
    print("\n=== Column Descriptions ===")
    for col in df.columns:
        if col in data_dict["dictionary"]:
            print(f"{col}: {data_dict['dictionary'][col]}")

    # Display missing values
    print("\n=== Missing Values ===")
    for col, count in basic_info["missing_values"].items():
        if count > 0:
            percentage = (count / len(df)) * 100
            print(f"{col}: {count} ({percentage:.2f}%)")

    # Analyze target variable
    target_stats = analyze_target_variable(df, target_col=target_col)
    print("\n=== Target Variable Analysis ===")
    print("Class distribution:")
    for label, count in target_stats["distribution"].items():
        percentage = target_stats["distribution_percentage"][label] * 100
        print(f"Class {label}: {count} ({percentage:.2f}%)")

    # Get variable types
    var_types = get_variable_types(df, excluded_cols=excluded_cols)
    print("\n=== Variable Types ===")
    print(
        f"Continuous variables ({len(var_types['continuous'])}):",
        ", ".join(var_types["continuous"]),
    )
    print(
        f"\nCategorical variables ({len(var_types['categorical'])}):",
        ", ".join(var_types["categorical"]),
    )


if __name__ == "__main__":
    main()
