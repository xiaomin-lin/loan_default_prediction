"""
Main EDA script that orchestrates the analysis workflow.
Author: Xiaomin Lin
Date: 2025-01-18
"""

from pathlib import Path
from typing import List, Optional

from src.eda.data_inspection import load_data, transform_data
from src.utils.data_utils import clean_data, get_variable_types
from src.utils.file_utils import create_output_dirs
from src.utils.stats_utils import (
    calculate_correlations_with_target,
    descriptive_statistics,
    rank_features_by_impact,
)
from src.visualization.plots import (
    plot_boxplots,
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_correlation_rankings,
    plot_distributions,
    plot_feature_importance,
    plot_target_relationships,
)


def run_eda(
    output_dir: Path,
    excluded_cols: Optional[List[str]] = None,
    target_col: str = "bad_flag",
) -> None:
    """Main function to run the complete EDA workflow."""
    # Create output directories
    plots_dir = create_output_dirs(output_dir)

    # Load and prepare data
    df, _ = load_data()
    df_transformed = transform_data(df)

    df_clean, cleaning_stats = clean_data(
        df=df_transformed,
        id_col="member_id",
        target_col=target_col,
        excluded_cols=excluded_cols,
        dedup_strategy="first",
        verbose=True,
    )

    print(f"Excluding columns from analysis: {excluded_cols}")

    # Print df_clean head and shape
    print(f"df_clean head: {df_clean.head()}")
    print(f"df_clean shape: {df_clean.shape}")

    # Print variable types
    var_types = get_variable_types(df_clean, excluded_cols)
    print(
        f"Continuous variables: {var_types['continuous'] if var_types['continuous'] else 'None'}"
    )
    print(
        f"Categorical variables: {var_types['categorical'] if var_types['categorical'] else 'None'}"
    )

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

    # Plot target relationships
    print("\nGenerating target relationship plots...")
    plot_target_relationships(df_clean, plots_dir, excluded_cols, target_col)

    # Rank features
    impact_results = rank_features_by_impact(df_clean, excluded_cols, target_col)

    # Create visualizations
    plot_feature_importance(impact_results, plots_dir)

    # Print summary with more details
    print("\nFeature Impact Analysis:")
    print("\nTop 10 Most Important Features:")
    print(impact_results.head(10).to_string(index=False))

    print("\nFeature Impact Distribution:")
    print(impact_results["impact_level"].value_counts().to_string())

    # Calculate correlations
    correlation_results = calculate_correlations_with_target(
        df_clean, target_col, excluded_cols
    )
    print(f"Correlation results: {correlation_results}")
    # Create visualizations for correlations
    plot_correlation_rankings(correlation_results, plots_dir)

    # print("\nTop 10 Features by Correlation with Target:")
    print(correlation_results.to_string())
    print("\nNote:")
    print("- For continuous features: Point-biserial correlation is shown")
    print("- For categorical features: Cramer's V is shown")
    print("- All correlation coefficients are in range [0, 1]")

    # Additional analysis as needed
    # ...


if __name__ == "__main__":
    output_dir = Path("reports/eda")
    excluded_cols = ["desc", "member_id", "id", "application_approved_flag"]
    target_col = "bad_flag"
    run_eda(output_dir, excluded_cols, target_col)
