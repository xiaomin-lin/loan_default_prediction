"""
Initial data inspection script for loan default prediction project.
Author: Xiaomin Lin
Date: 2025-01-17
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


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

    # Load training data, skipping the first row and using the second row as header
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


def get_basic_info(df: pd.DataFrame, excluded_cols: Optional[List[str]] = None) -> Dict:
    """
    Get basic information about the dataset
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        dict: Basic dataset information
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


def analyze_target_variable(df: pd.DataFrame, target_col: str = "bad_flag") -> Dict:
    """
    Analyze the target variable
    Args:
        df: pandas DataFrame
        target_col: Name of target column (default: "bad_flag")
    Returns:
        dict: Target variable statistics
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


def check_duplicate_member_ids(
    df: pd.DataFrame, member_id_col: str = "member_id"
) -> None:
    """
    Check for duplicate member_ids in the dataset
    Args:
        df: pandas DataFrame
        member_id_col: Name of member ID column (default: "member_id")
    Returns:
        None
    """
    duplicate_members = df[df.duplicated(subset=member_id_col, keep=False)]

    if not duplicate_members.empty:
        print("\n=== Duplicate Member IDs ===")
        print(duplicate_members[[member_id_col]].drop_duplicates())
        print(
            f"Total duplicate {member_id_col}s: {duplicate_members[member_id_col].nunique()}"
        )
    else:
        print(f"\nNo duplicate {member_id_col}s found.")


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

    # Transforming the specified columns if they exist in columns_to_transform
    if "id" in columns_to_transform:
        df_transformed["id"] = df_transformed["id"].astype(str)
    if "member_id" in columns_to_transform:
        df_transformed["member_id"] = df_transformed["member_id"].astype(str)
    if "loan_amnt" in columns_to_transform:
        df_transformed["loan_amnt"] = df_transformed["loan_amnt"].astype(int)
    if "term" in columns_to_transform:
        df_transformed["term"] = df_transformed["term"].astype(str)
    if "emp_length" in columns_to_transform:
        df_transformed["emp_length"] = df_transformed["emp_length"].astype(str)
    if "int_rate" in columns_to_transform:
        df_transformed["int_rate"] = (
            df_transformed["int_rate"].str.replace("%", "").astype(float) / 100.0
        )
    if "revol_util" in columns_to_transform:
        df_transformed["revol_util"] = (
            df_transformed["revol_util"].str.replace("%", "").astype(float) / 100.0
        )

    # Transform 'emp_length' from string to integer if it exists
    if "emp_length" in columns_to_transform:

        def parse_emp_length(emp_str):
            if pd.isnull(emp_str):
                return None
            # Extract the numeric part
            num = "".join(filter(str.isdigit, emp_str))
            return int(num) if num else None

        df_transformed["emp_length"] = df_transformed["emp_length"].apply(
            parse_emp_length
        )

    return df_transformed


def main():
    """Main function to run initial data inspection"""
    # Define columns to exclude
    excluded_cols = ["desc", "member_id", "id"]  # Add any columns you want to exclude
    target_col = "bad_flag"

    # Load data
    print("Loading data and verifying columns...")
    df, data_dict = load_data(excluded_cols=excluded_cols)

    # Transform the data
    df = transform_data(df, excluded_cols=excluded_cols)

    print(df.head())

    # Get basic information
    basic_info = get_basic_info(df, excluded_cols=excluded_cols)
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
