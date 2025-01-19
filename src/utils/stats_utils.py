"""Statistical analysis utility functions."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.utils.data_utils import get_variable_types


def descriptive_statistics(df: pd.DataFrame, excluded_cols: list = None) -> None:
    """
    Generate and print descriptive statistics for the dataset.
    Args:
        df (pd.DataFrame): The input DataFrame
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)

    print("\n=== Descriptive Statistics ===")
    print("\n-- Continuous Variables --")
    print(df[var_types["continuous"]].describe())

    print("\n-- Categorical Variables --")
    print(df[var_types["categorical"]].describe())


def calculate_woe_iv(
    df: pd.DataFrame,
    feature: str,
    excluded_cols: Optional[List[str]] = None,
    target_col: str = "bad_flag",
    bins: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Weight of Evidence and Information Value for a feature.

    Args:
        df: Input DataFrame
        feature: Feature column name
        excluded_cols: List of columns to exclude from analysis
        target_col: Target variable name (default: 'bad_flag')
        bins: Number of bins for continuous variables (default: 10)

    Returns:
        Tuple of (WOE DataFrame, Information Value)
    """
    df_copy = df.copy()

    var_types = get_variable_types(df, excluded_cols)
    continuous_cols = var_types["continuous"]
    categorical_cols = var_types["categorical"]

    # Use predefined variable types
    if (
        feature == target_col
        or feature in excluded_cols
        or feature not in continuous_cols + categorical_cols
    ):
        raise ValueError(
            f"Feature {feature} not found in either continuous or categorical columns, or is the target variable, or should be excluded!"
        )

    if feature in continuous_cols and df[feature].nunique() > 20:
        try:
            df_copy["bin"] = pd.qcut(df[feature], bins, duplicates="drop")
        except ValueError:
            df_copy["bin"] = pd.cut(df[feature], bins, duplicates="drop")
    else:
        df_copy["bin"] = df[feature]

    # Calculate WOE and IV
    grouped = df_copy.groupby("bin")[target_col].agg(["count", "sum"])
    grouped["non_event"] = grouped["count"] - grouped["sum"]
    grouped["event"] = grouped["sum"]

    epsilon = 1e-10
    grouped["event_rate"] = (grouped["event"] + epsilon) / (
        grouped["event"].sum() + epsilon
    )
    grouped["non_event_rate"] = (grouped["non_event"] + epsilon) / (
        grouped["non_event"].sum() + epsilon
    )

    grouped["woe"] = np.log(grouped["non_event_rate"] / grouped["event_rate"])
    grouped["iv"] = (grouped["non_event_rate"] - grouped["event_rate"]) * grouped["woe"]

    grouped["woe"] = grouped["woe"].replace([np.inf, -np.inf], 0)
    grouped["iv"] = grouped["iv"].replace([np.inf, -np.inf], 0)

    total_iv = grouped["iv"].sum()

    return grouped, total_iv


def calculate_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculate Cramer's V statistic between two categorical variables."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2 = chi2_contingency(contingency)[0]
    n = len(df)
    min_dim = min(contingency.shape) - 1

    # Calculate Cramer's V
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    return v


def calculate_correlations_with_target(
    df: pd.DataFrame,
    target_col: str,
    excluded_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate correlations between features and target variable.

    Args:
        df: Input DataFrame
        target_col: Target variable name
        excluded_cols: List of columns to exclude

    Returns:
        DataFrame with correlation coefficients and methods used
    """
    correlations = {}
    correlation_methods = {}

    var_types = get_variable_types(df, excluded_cols)
    # Calculate point-biserial correlations for continuous variables
    for col in var_types["continuous"]:
        if col != target_col and (not excluded_cols or col not in excluded_cols):
            complete_cases = df[[col, target_col]].dropna()
            if len(complete_cases) > 0:
                try:
                    corr = complete_cases[col].corr(
                        complete_cases[target_col].astype("int")
                    )
                    correlations[col] = corr
                    correlation_methods[col] = "Point-biserial"
                except Exception as e:
                    print(f"Could not calculate correlation for {col}: {str(e)}")

    # Calculate Cramer's V for categorical variables
    for col in var_types["categorical"]:
        if col != target_col and (not excluded_cols or col not in excluded_cols):
            try:
                complete_cases = df[[col, target_col]].dropna()
                if len(complete_cases) > 0:
                    v = calculate_cramers_v(complete_cases, col, target_col)
                    correlations[col] = v
                    correlation_methods[col] = "Cramer's V"
            except Exception as e:
                print(f"Could not calculate correlation for {col}: {str(e)}")

    # Create a DataFrame with both correlations and methods
    results = pd.DataFrame(
        {"correlation": correlations, "method": correlation_methods}
    ).sort_values("correlation", ascending=False)

    return results


def rank_features_by_impact(
    df: pd.DataFrame,
    excluded_cols: Optional[List[str]] = None,
    target_col: str = "bad_flag",
) -> pd.DataFrame:
    """
    Rank all features by their impact on target using Information Value.

    Args:
        df: Input DataFrame
        excluded_cols: List of columns to exclude
        target_col: Target variable name

    Returns:
        DataFrame with features ranked by their Information Value
    """
    if excluded_cols is None:
        excluded_cols = []

    # Create a copy with only labeled cases (where target is not None/NaN)
    df_labeled = df[df[target_col].notna()].copy()

    # Convert target to numeric type for calculations
    df_labeled[target_col] = df_labeled[target_col].astype(int)

    # Get features to analyze
    features = [col for col in df.columns if col not in excluded_cols + [target_col]]

    # Calculate IV for each feature
    feature_impact = []
    for feature in features:
        try:
            # Only use cases where both feature and target are not null
            df_feature = df_labeled[df_labeled[feature].notna()][[feature, target_col]]
            if len(df_feature) > 0:  # Only process if we have valid cases
                _, iv = calculate_woe_iv(df_feature, feature, excluded_cols, target_col)
                feature_impact.append(
                    {
                        "feature": feature,
                        "iv": iv,
                        "impact_level": pd.cut(
                            [iv],
                            bins=[0, 0.02, 0.1, 0.3, np.inf],
                            labels=["Weak", "Medium", "Strong", "Very Strong"],
                        )[0],
                    }
                )
        except Exception as e:
            print(f"Error processing {feature}: {str(e)}")

    # Create results DataFrame
    if feature_impact:
        impact_df = pd.DataFrame(feature_impact).sort_values("iv", ascending=False)
    else:
        impact_df = pd.DataFrame(columns=["feature", "iv", "impact_level"])

    return impact_df
