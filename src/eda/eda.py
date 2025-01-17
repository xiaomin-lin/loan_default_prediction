"""
Exploratory Data Analysis (EDA) script for loan default prediction project.
Author: [Your Name]
Date: 2025-01-19
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_inspection import (
    check_duplicate_member_ids,
    get_variable_types,
    load_data,
    transform_data,
)
from data_quality import clean_dataset

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
    continuous_cols = var_types["continuous"]

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
    df: pd.DataFrame, plots_dir: Path, excluded_cols: list = None
) -> None:
    """
    Analyze the relationship between features and the target variable.
    Args:
        df (pd.DataFrame): The input DataFrame
        plots_dir (Path): Directory to save the plots
        excluded_cols (list): List of columns to exclude from analysis
    """
    var_types = get_variable_types(df, excluded_cols)
    target = "bad_flag"

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


def main():
    """Main function to perform EDA"""
    # Define output directories
    root_dir = Path.cwd()
    output_dir = root_dir / "reports" / "eda"
    plots_dir = create_output_dirs(output_dir)

    # Load and clean data
    print("Loading and cleaning data...")
    df_raw, data_dict = load_data()
    df_clean, cleaning_stats = clean_dataset(df_raw)

    # Transform data types
    df_clean = transform_data(df_clean)

    # Check for duplicate member IDs
    check_duplicate_member_ids(df_clean)

    # Define columns to exclude from analysis
    excluded_cols = ["desc", "id", "member_id"]  # Add any other columns to exclude
    print(f"Excluding columns from analysis: {excluded_cols}")

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
    analyze_target_relationship(df_clean, plots_dir, excluded_cols)

    print("\nEDA completed. All plots are saved in the reports/eda/plots directory.")


if __name__ == "__main__":
    main()
