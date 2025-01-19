"""Visualization functions for EDA."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.data_utils import get_variable_types

# Configure visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def plot_distributions(
    df: pd.DataFrame, plots_dir: Path, excluded_cols: list = None
) -> None:
    """Plot distributions of continuous variables."""
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
    continuous_cols = var_types["continuous"]

    plt.figure(figsize=(15, 12))
    corr = df[continuous_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = plots_dir / "correlation_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved correlation heatmap at {plot_path}")


def plot_target_relationships(
    df: pd.DataFrame,
    plots_dir: Path,
    excluded_cols: list = None,
    target: str = "bad_flag",
) -> None:
    """Plot relationships between features and target."""
    var_types = get_variable_types(df, excluded_cols)
    # Analyze continuous variables
    continuous_cols = [col for col in var_types["continuous"] if col != target]
    # Continuous variables
    for col in continuous_cols:
        if col != target:
            plt.figure()
            sns.boxplot(x=target, y=col, data=df)
            plt.title(f"{col} vs {target}")
            plt.tight_layout()
            plot_path = plots_dir / f"{col}_vs_{target}_boxplot.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved boxplot of {col} vs {target} at {plot_path}")

    # Analyze categorical variables
    categorical_cols = [col for col in var_types["categorical"] if col != target]
    for col in categorical_cols:
        if col != target and df[col].nunique() <= 20:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=col, hue=target, data=df)
            plt.title(f"{col} Distribution by {target}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = plots_dir / f"{col}_distribution_by_{target}_countplot.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved count plot of {col} by {target} at {plot_path}")


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

    # # Distribution of correlation methods
    # plt.figure(figsize=(10, 6))
    # method_counts = correlation_df["method"].value_counts()
    # sns.barplot(x=method_counts.index, y=method_counts.values)
    # plt.title("Distribution of Correlation Methods")
    # plt.xlabel("Correlation Method")
    # plt.ylabel("Count of Features")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(plots_dir / "correlation_methods_distribution.png")
    # plt.close()
