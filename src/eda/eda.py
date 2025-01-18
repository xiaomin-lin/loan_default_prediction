"""
Exploratory Data Analysis (EDA) script for loan default prediction project.
Author: [Your Name]
Date: 2025-01-19
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_inspection import (
    check_duplicate_member_ids,
    get_variable_types,
    load_data,
    transform_data,
)
from data_quality import clean_dataset
from scipy.stats import chi2_contingency

# Configure visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def create_output_dirs(output_dir: Path):
    """
    Create directories to save EDA plots.
    Args:
        output_dir (Path): Path to the output directory
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {output_dir}")
    plots_dir = output_dir / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory for plots: {plots_dir}")
    else:
        # Clear existing plots
        for file in plots_dir.iterdir():
            if file.is_file():
                file.unlink()
        print(f"Cleared existing plots in: {plots_dir}")
    return plots_dir


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


def plot_distributions(
    df: pd.DataFrame, plots_dir: Path, excluded_cols: list = None
) -> None:
    """
    Plot distributions of continuous variables.
    Args:
        df (pd.DataFrame): The input DataFrame
        plots_dir (Path): Directory to save the plots
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)
    continuous_cols = var_types["continuous"]

    for col in continuous_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plot_path = plots_dir / f"{col}_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved distribution plot for {col} at {plot_path}")


def plot_boxplots(
    df: pd.DataFrame, plots_dir: Path, excluded_cols: list = None
) -> None:
    """
    Plot boxplots for continuous variables to identify outliers.
    Args:
        df (pd.DataFrame): The input DataFrame
        plots_dir (Path): Directory to save the plots
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)
    continuous_cols = var_types["continuous"]

    for col in continuous_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plot_path = plots_dir / f"{col}_boxplot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved boxplot for {col} at {plot_path}")


def plot_categorical_counts(
    df: pd.DataFrame, plots_dir: Path, excluded_cols: list = None
) -> None:
    """
    Plot count plots for categorical variables.
    Args:
        df (pd.DataFrame): The input DataFrame
        plots_dir (Path): Directory to save the plots
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)
    categorical_cols = var_types["categorical"]

    for col in categorical_cols:
        if len(df[col].unique()) > 50:  # Skip if too many unique values
            print(f"Skipping {col} due to high cardinality")
            continue

        plt.figure(figsize=(12, 6))
        value_counts = df[col].value_counts()
        if len(value_counts) > 20:  # Limit to top 20 categories
            value_counts = value_counts.head(20)

        sns.barplot(x=value_counts.values, y=value_counts.index)
        plt.title(f"Count Plot of {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plot_path = plots_dir / f"{col}_countplot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved count plot for {col} at {plot_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame, plots_dir: Path, excluded_cols: list = None
) -> None:
    """
    Plot a correlation heatmap for continuous variables.
    Args:
        df (pd.DataFrame): The input DataFrame
        plots_dir (Path): Directory to save the plot
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)

    # Only use numeric columns for correlation
    continuous_cols = (
        df[var_types["continuous"]].select_dtypes(include=["int64", "float64"]).columns
    )

    plt.figure(figsize=(15, 12))
    corr = df[continuous_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plot_path = plots_dir / "correlation_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved correlation heatmap at {plot_path}")


def analyze_target_relationship(
    df: pd.DataFrame,
    plots_dir: Path,
    excluded_cols: list = None,
    target: str = "bad_flag",
) -> None:
    """
    Analyze the relationship between features and the target variable.
    Args:
        df (pd.DataFrame): The input DataFrame
        plots_dir (Path): Directory to save the plots
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)
    print(var_types["categorical"])
    # Analyze continuous variables
    continuous_cols = [col for col in var_types["continuous"] if col != target]
    for col in continuous_cols:
        plt.figure()
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f"{col} vs {target}")
        plt.xlabel(target)
        plt.ylabel(col)
        plt.tight_layout()
        plot_path = plots_dir / f"{col}_vs_{target}_boxplot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved boxplot of {col} vs {target} at {plot_path}")

    # Analyze categorical variables
    categorical_cols = [col for col in var_types["categorical"] if col != target]
    for col in categorical_cols:
        if len(df[col].unique()) > 20:  # Skip if too many categories
            print(f"Skipping {col} vs {target} plot due to high cardinality")
            continue

        plt.figure(figsize=(12, 6))
        sns.countplot(
            x=col, hue=target, data=df, order=df[col].value_counts().index[:20]
        )
        plt.title(f"{col} Distribution by {target}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title=target)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = plots_dir / f"{col}_distribution_by_{target}_countplot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved count plot of {col} by {target} at {plot_path}")


def calculate_woe_iv(
    df: pd.DataFrame, feature: str, target: str = "bad_flag", bins: int = 10
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate Weight of Evidence and Information Value for a feature.

    Args:
        df: Input DataFrame
        feature: Feature column name
        target: Target variable name (default: 'bad_flag')
        bins: Number of bins for continuous variables (default: 10)

    Returns:
        Tuple of (WOE DataFrame, Information Value)
    """
    # Create a copy to avoid modifying original DataFrame
    df_copy = df.copy()

    # Handle continuous vs categorical variables
    if df[feature].dtype in ["float64", "int64"] and df[feature].nunique() > 20:
        try:
            # For continuous variables, create equal-frequency bins
            df_copy["bin"] = pd.qcut(df[feature], bins, duplicates="drop")
        except ValueError:
            # If qcut fails (e.g., too many identical values), try cut
            df_copy["bin"] = pd.cut(df[feature], bins, duplicates="drop")
    else:
        # For categorical variables, use categories as is
        df_copy["bin"] = df[feature]

    # Calculate WOE and IV
    grouped = df_copy.groupby("bin")[target].agg(["count", "sum"])
    grouped["non_event"] = grouped["count"] - grouped["sum"]
    grouped["event"] = grouped["sum"]

    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    grouped["event_rate"] = (grouped["event"] + epsilon) / (
        grouped["event"].sum() + epsilon
    )
    grouped["non_event_rate"] = (grouped["non_event"] + epsilon) / (
        grouped["non_event"].sum() + epsilon
    )

    # Calculate WOE and IV
    grouped["woe"] = np.log(grouped["non_event_rate"] / grouped["event_rate"])
    grouped["iv"] = (grouped["non_event_rate"] - grouped["event_rate"]) * grouped["woe"]

    # Clean up infinite values
    grouped["woe"] = grouped["woe"].replace([np.inf, -np.inf], 0)
    grouped["iv"] = grouped["iv"].replace([np.inf, -np.inf], 0)

    total_iv = grouped["iv"].sum()

    return grouped, total_iv


def rank_features_by_impact(
    df: pd.DataFrame,
    excluded_cols: Optional[List[str]] = None,
    target: str = "bad_flag",
) -> pd.DataFrame:
    """
    Rank all features by their impact on target using Information Value.

    Args:
        df: Input DataFrame
        excluded_cols: List of columns to exclude
        target: Target variable name

    Returns:
        DataFrame with features ranked by their Information Value
    """
    if excluded_cols is None:
        excluded_cols = []

    # Create a copy with only labeled cases (where target is not None/NaN)
    df_labeled = df[df[target].notna()].copy()

    # Convert target to numeric type for calculations
    df_labeled[target] = df_labeled[target].astype(int)

    # Get features to analyze
    features = [col for col in df.columns if col not in excluded_cols + [target]]

    # Calculate IV for each feature
    feature_impact = []
    for feature in features:
        try:
            # Only use cases where both feature and target are not null
            df_feature = df_labeled[df_labeled[feature].notna()][[feature, target]]
            if len(df_feature) > 0:  # Only process if we have valid cases
                _, iv = calculate_woe_iv(df_feature, feature, target)
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
        results = pd.DataFrame(feature_impact)
        results = results.sort_values("iv", ascending=False)
    else:
        results = pd.DataFrame(columns=["feature", "iv", "impact_level"])

    return results


def plot_feature_importance(impact_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Create visualizations for feature importance analysis.

    Args:
        impact_df: DataFrame with feature impact results
        plots_dir: Directory to save plots
    """
    # Bar plot of Information Values
    plt.figure(figsize=(12, 6))
    sns.barplot(data=impact_df, x="iv", y="feature")
    plt.title("Feature Importance by Information Value")
    plt.xlabel("Information Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_importance_iv.png")
    plt.close()

    # Distribution of impact levels
    plt.figure(figsize=(10, 6))
    impact_counts = impact_df["impact_level"].value_counts()
    sns.barplot(x=impact_counts.index, y=impact_counts.values)
    plt.title("Distribution of Feature Impact Levels")
    plt.xlabel("Impact Level")
    plt.ylabel("Count of Features")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_impact_distribution.png")
    plt.close()


def plot_correlation_rankings(correlation_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Create visualizations for correlation analysis results.

    Args:
        correlation_df: DataFrame with correlation results (contains 'correlation' and 'method' columns)
        plots_dir: Directory to save plots
    """
    # Bar plot of correlation values
    plt.figure(figsize=(12, 6))
    # Create color mapping for different correlation methods
    method_colors = {"Point-biserial": "skyblue", "Cramer's V": "lightgreen"}

    # Create bar plot with different colors for different methods
    ax = plt.gca()
    bars = plt.barh(
        y=correlation_df.index,
        width=correlation_df["correlation"],
        color=[method_colors[method] for method in correlation_df["method"]],
    )

    plt.title("Feature Correlations with Target")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Feature")

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=method)
        for method, color in method_colors.items()
    ]
    ax.legend(handles=legend_elements, title="Correlation Method")

    plt.tight_layout()
    plt.savefig(plots_dir / "feature_correlations.png")
    plt.close()

    # Distribution of correlation methods
    plt.figure(figsize=(10, 6))
    method_counts = correlation_df["method"].value_counts()
    sns.barplot(x=method_counts.index, y=method_counts.values)
    plt.title("Distribution of Correlation Methods")
    plt.xlabel("Correlation Method")
    plt.ylabel("Count of Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_methods_distribution.png")
    plt.close()


def analyze_feature_impact(
    df: pd.DataFrame,
    plots_dir: Path,
    excluded_cols: Optional[List[str]] = None,
    target: str = "bad_flag",
) -> None:
    """
    Analyze and visualize feature impact on target variable.

    Args:
        df: Input DataFrame
        plots_dir: Directory to save plots
        excluded_cols: List of columns to exclude
    """
    # Rank features
    impact_results = rank_features_by_impact(df, excluded_cols, target)

    # Create visualizations
    plot_feature_importance(impact_results, plots_dir)

    # Print summary with more details
    print("\nFeature Impact Analysis:")
    print("\nTop 10 Most Important Features:")
    print(impact_results.head(10).to_string(index=False))

    print("\nFeature Impact Distribution:")
    print(impact_results["impact_level"].value_counts().to_string())

    # Calculate correlations
    var_types = get_variable_types(df, excluded_cols)
    correlation_results = calculate_correlations_with_target(
        df, target, var_types, excluded_cols
    )

    # Create visualizations for correlations
    plot_correlation_rankings(correlation_results, plots_dir)

    # print("\nTop 10 Features by Correlation with Target:")
    print(correlation_results.to_string())
    print("\nNote:")
    print("- For continuous features: Point-biserial correlation is shown")
    print("- For categorical features: Cramer's V is shown")
    print("- All correlation coefficients are in range [0, 1]")


def calculate_correlations_with_target(
    df: pd.DataFrame,
    target: str,
    var_types: dict,
    excluded_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate correlations between features and target variable using appropriate methods.

    Args:
        df: Input DataFrame
        target: Name of target variable
        var_types: Dictionary containing 'continuous' and 'categorical' variable lists
        excluded_cols: List of columns to exclude from analysis

    Returns:
        pd.DataFrame: DataFrame with correlation coefficients and methods used
    """
    continuous_cols = var_types["continuous"]
    categorical_cols = var_types["categorical"]

    correlations = {}
    correlation_methods = {}  # Track which method was used for each feature

    # Calculate point-biserial correlations for continuous variables
    for col in continuous_cols:
        if col != target:
            complete_cases = df[[col, target]].dropna()
            if len(complete_cases) > 0:
                try:
                    corr = complete_cases[col].corr(
                        complete_cases[target].astype("int")
                    )
                    correlations[col] = corr
                    correlation_methods[col] = "Point-biserial"
                except Exception as e:
                    print(f"Could not calculate correlation for {col}: {str(e)}")

    # Calculate appropriate correlation measure for categorical variables
    for col in categorical_cols:
        if col != target:
            try:
                complete_cases = df[[col, target]].dropna()
                if len(complete_cases) > 0:
                    # Use Cramer's V for all categorical variables
                    v = calculate_cramers_v(complete_cases, col, target)
                    correlations[col] = v
                    correlation_methods[col] = "Cramer's V"
            except Exception as e:
                print(f"Could not calculate correlation for {col}: {str(e)}")

    # Create a DataFrame with both correlations and methods
    results = pd.DataFrame(
        {"correlation": correlations, "method": correlation_methods}
    ).sort_values("correlation", ascending=False)

    return results


def calculate_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculate Cramer's V statistic between two categorical variables."""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2 = chi2_contingency(contingency)[0]
    n = len(df)
    min_dim = min(contingency.shape) - 1

    # Calculate Cramer's V
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    return v


def main():
    """Main function to perform EDA"""
    # Define output directories
    root_dir = Path.cwd()
    output_dir = root_dir / "reports" / "eda"
    plots_dir = create_output_dirs(output_dir)

    # Load and clean data
    print("Loading and cleaning data...")
    df_raw, data_dict = load_data()
    # Transform data types
    df_transformed = transform_data(df_raw)
    # Clean data
    df_clean, cleaning_stats = clean_dataset(df_transformed)

    # Check for duplicate member IDs
    check_duplicate_member_ids(df_clean)

    # Define columns to exclude from analysis
    excluded_cols = [
        "desc",
        "id",
        "member_id",
        "application_approved_flag",
    ]  # Add any other columns to exclude
    print(f"Excluding columns from analysis: {excluded_cols}")

    # target variable
    target = "bad_flag"

    # Descriptive statistics
    descriptive_statistics(df_clean, excluded_cols)

    # Plot distributions
    print("\nGenerating distribution plots...")
    plot_distributions(df_clean, plots_dir, excluded_cols)

    # Plot boxplots
    print("\nGenerating boxplots for continuous variables...")
    plot_boxplots(df_clean, plots_dir, excluded_cols)

    # Plot categorical counts
    print("\nGenerating count plots for categorical variables...")
    plot_categorical_counts(df_clean, plots_dir, excluded_cols)

    # Plot correlation heatmap
    print("\nGenerating correlation heatmap...")
    plot_correlation_heatmap(df_clean, plots_dir, excluded_cols)

    # Analyze target relationships
    print("\nAnalyzing relationships between features and target variable...")
    analyze_target_relationship(df_clean, plots_dir, excluded_cols, target)

    # Analyze feature impact
    print("\nAnalyzing feature impact on target variable...")
    analyze_feature_impact(df_clean, plots_dir, excluded_cols, target)

    print("\nEDA completed. All plots are saved in the reports/eda/plots directory.")


if __name__ == "__main__":
    main()
