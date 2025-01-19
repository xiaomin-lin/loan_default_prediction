"""
Feature engineering pipeline for loan default prediction.
Author: Xiaomin Lin
Date: 2025-01-19
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.features.engineers.desc_engineer import DescriptionFeatureEngineer
from src.features.engineers.loan_engineer import LoanFeatureEngineer


def engineer_features(
    df: pd.DataFrame,
    desc_engineer: Optional[DescriptionFeatureEngineer] = None,
    loan_engineer: Optional[LoanFeatureEngineer] = None,
    excluded_cols: Optional[List[str]] = None,
    target_col: str = "bad_flag",
) -> Tuple[pd.DataFrame, Dict, DescriptionFeatureEngineer, LoanFeatureEngineer]:
    """
    Engineer features for loan default prediction.

    Args:
        df: Input DataFrame
        desc_engineer: Optional pre-fitted description engineer
        loan_engineer: Optional pre-fitted loan engineer
        excluded_cols: Columns to exclude
        target_col: Target variable column

    Returns:
        Tuple of:
        - DataFrame with engineered features
        - Engineering statistics
        - Fitted description engineer
        - Fitted loan engineer
    """
    engineering_stats = {"original_shape": df.shape}
    df_engineered = df.copy()

    # Initialize/use engineers
    if loan_engineer is None:
        print("Fitting loan engineer...")
        loan_engineer = LoanFeatureEngineer()
        loan_engineer.fit(df_engineered)

    if desc_engineer is None and "desc" in df_engineered.columns:
        print("Fitting description engineer...")
        desc_engineer = DescriptionFeatureEngineer()
        desc_engineer.fit(df_engineered["desc"])

    # Transform features
    print("Engineering loan features...")
    df_engineered = loan_engineer.transform(df_engineered)

    if desc_engineer is not None and "desc" in df_engineered.columns:
        print("Engineering description features...")
        desc_features = desc_engineer.transform(df_engineered["desc"])
        df_engineered = pd.concat(
            [df_engineered.drop(columns=["desc"]), desc_features], axis=1
        )

    # Update statistics
    engineering_stats.update(
        {
            "final_shape": df_engineered.shape,
            "new_features": list(set(df_engineered.columns) - set(df.columns)),
            "feature_count": {
                "original": len(df.columns),
                "engineered": len(df_engineered.columns),
                "new": len(df_engineered.columns) - len(df.columns),
            },
        }
    )

    return df_engineered, engineering_stats, desc_engineer, loan_engineer


def save_engineers(
    desc_engineer: DescriptionFeatureEngineer, loan_engineer: LoanFeatureEngineer
):
    """Save the fitted engineers to disk."""
    # Create directories if they don't exist
    save_dir = Path("models/trained/feature_engineers")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save engineers
    joblib.dump(desc_engineer, save_dir / "desc_engineer.pkl")
    joblib.dump(loan_engineer, save_dir / "loan_engineer.pkl")
    print(f"Engineers saved to {save_dir}")


def load_engineers(save_dir: Path = Path("models/trained/feature_engineers")):
    """Load the fitted engineers from disk."""
    desc_engineer = joblib.load(save_dir / "desc_engineer.pkl")
    loan_engineer = joblib.load(save_dir / "loan_engineer.pkl")
    return desc_engineer, loan_engineer


def main():
    """Main function demonstrating the feature engineering pipeline."""
    from src.eda.data_inspection import load_data, transform_data
    from src.utils.data_utils import clean_data

    # Load and prepare training data
    print("Loading training data...")
    df_train, _ = load_data()
    df_train = transform_data(df_train)

    # Clean data
    df_train_clean, _ = clean_data(
        df=df_train,
        id_col="member_id",
        target_col="bad_flag",
        excluded_cols=["id", "application_approved_flag", "member_id"],
        dedup_strategy="first",
        verbose=True,
    )

    # Engineer features and get fitted engineers
    print("\nEngineering features for training data...")
    df_train_engineered, stats, desc_engineer, loan_engineer = engineer_features(
        df_train_clean
    )

    # Example: Using the fitted engineers on new data
    print("\nExample: Engineering features for test data...")
    df_test = df_train_clean.head()  # Simulated test data
    df_test_engineered, _, _, _ = engineer_features(
        df_test, desc_engineer=desc_engineer, loan_engineer=loan_engineer
    )

    print("\nFeature Engineering Summary:")
    print(f"Training shape: {df_train_engineered.shape}")
    print(f"Test shape: {df_test_engineered.shape}")
    print("\nNew features created:")
    for feature in stats["new_features"]:
        print(f"- {feature}")
    # print the first 5 rows of the engineered dataframe, only containing the engineered features as well as the original description column
    print(
        "\nFirst 5 rows of the engineered dataframe (only containing engineered features and the original description column):"
    )
    print(df_train_engineered[stats["new_features"]].head())
    # Print df_train_engineered's columns
    print("\nColumns of df_train_engineered:")
    print(df_train_engineered.columns)

    # Save the engineers
    save_engineers(desc_engineer, loan_engineer)


if __name__ == "__main__":
    main()
