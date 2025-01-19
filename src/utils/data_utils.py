from typing import Dict, List, Optional, Tuple

import pandas as pd


def check_duplication(
    df: pd.DataFrame, excluded_cols: Optional[List[str]] = None, verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Check for duplicate rows in a DataFrame, excluding specified columns, and return the duplicate entries
    with all their columns for comparison.

    Args:
        df: Input pandas DataFrame
        excluded_cols: List of columns to exclude from duplication check (default: None)
        verbose: Whether to print duplicate information (default: True)

    Returns:
        Optional[pd.DataFrame]: DataFrame containing all rows with duplicate values, or None if no duplicates found
    """
    if excluded_cols is not None:
        # Create a copy of the DataFrame excluding the specified columns
        df_to_check = df.drop(columns=excluded_cols, errors="ignore")
    else:
        df_to_check = df.copy()

    # Check for duplicates
    duplicate_mask = df_to_check.duplicated(keep=False)
    duplicates_df = df[duplicate_mask].sort_values(by=list(df.columns))

    if duplicates_df.empty:
        if verbose:
            print("\nNo duplications found in the DataFrame.")
        return None

    if verbose:
        print("\n=== Duplicate Analysis ===")
        print(f"Number of duplicate rows: {duplicates_df.shape[0]}")
        print(f"Total rows with duplicates: {duplicates_df.shape[0]}")

    return duplicates_df


def get_duplicates_by_column(
    df: pd.DataFrame, column: str, verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Check for duplicates in a DataFrame based on a specific column and return the duplicate entries
    with all their columns for comparison.

    Args:
        df: Input pandas DataFrame
        column: Column name to check for duplicates
        verbose: Whether to print duplicate information (default: True)

    Returns:
        Optional[pd.DataFrame]: DataFrame containing all rows with duplicate values in the specified column,
                              or None if no duplicates found
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    duplicate_mask = df.duplicated(subset=column, keep=False)
    duplicates_df = df[duplicate_mask].sort_values(by=column)

    if duplicates_df.empty:
        if verbose:
            print(f"\nNo duplications found based on column '{column}'")
        return None

    if verbose:
        print(f"\n=== Duplicate Analysis for column '{column}' ===")
        print(f"Number of duplicate values: {duplicates_df[column].nunique()}")
        print(f"Total rows with duplicates: {len(duplicates_df)}")

    return duplicates_df


def get_basic_info(df: pd.DataFrame, excluded_cols: Optional[List[str]] = None) -> Dict:
    """
    Get basic information about the dataset
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        dict: Basic dataset information including shape, columns, dtypes, and missing values
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


def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Analyze the target variable distribution and missing values
    Args:
        df: pandas DataFrame
        target_col: Name of target column
    Returns:
        dict: Target variable statistics including distribution and missing values
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
    Categorize variables by their types (continuous vs categorical)
    Args:
        df: pandas DataFrame
        excluded_cols: List of columns to exclude from analysis (default: None)
    Returns:
        dict: Variables categorized by type
    """
    if excluded_cols is None:
        excluded_cols = []

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


def impute_continuous_features(
    df: pd.DataFrame,
    continuous_cols: List[str],
    excluded_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    method: str = "median",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for continuous features
    Args:
        df: Input DataFrame
        continuous_cols: List of continuous column names
        excluded_cols: List of columns to exclude from imputation (default: None)
        target_col: Name of target column to exclude (default: None)
        method: Imputation method ('median' or 'mean', default: 'median')
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)
    stats = {"imputed_columns": {}}

    cols_to_impute = continuous_cols
    if excluded_cols:
        cols_to_impute = [col for col in cols_to_impute if col not in excluded_cols]
    if target_col:
        cols_to_impute = [col for col in cols_to_impute if col != target_col]

    for col in cols_to_impute:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            if method == "mean":
                imputed_value = processed_df[col].mean()
            else:  # default to median
                imputed_value = processed_df[col].median()

            processed_df.loc[processed_df[col].isnull(), col] = imputed_value
            stats["imputed_columns"][col] = {
                "missing_count": missing_count,
                "imputed_value": imputed_value,
                "imputation_method": method,
            }

    return processed_df, stats


def impute_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    excluded_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    method: str = "mode",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for categorical features
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        excluded_cols: List of columns to exclude from imputation (default: None)
        target_col: Name of target column to exclude (default: None)
        method: Imputation method (currently only 'mode' supported)
    Returns:
        tuple: (processed_df, stats)
    """
    processed_df = df.copy(deep=True)
    stats = {"imputed_columns": {}}

    cols_to_impute = categorical_cols
    if excluded_cols:
        cols_to_impute = [col for col in cols_to_impute if col not in excluded_cols]
    if target_col:
        cols_to_impute = [col for col in cols_to_impute if col != target_col]

    for col in cols_to_impute:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            mode_value = processed_df[col].mode()[0]
            processed_df.loc[processed_df[col].isnull(), col] = mode_value
            stats["imputed_columns"][col] = {
                "missing_count": missing_count,
                "imputed_value": mode_value,
                "imputation_method": method,
            }

    return processed_df, stats


def handle_continuous_outliers(
    df: pd.DataFrame,
    continuous_cols: List[str],
    threshold: float = 1.5,
    excluded_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Identify outliers in continuous features based on the IQR method.
    Args:
        df: Input DataFrame
        continuous_cols: List of continuous column names
        threshold: IQR multiplier for outlier detection (default: 1.5)
        excluded_cols: List of columns to exclude from outlier detection (default: None)
        target_col: Name of target column to exclude (default: None)
    Returns:
        tuple: (processed_df, stats)
    """
    stats = {"outliers": {}}
    processed_df = df.copy()

    cols_to_analyze = continuous_cols
    if excluded_cols:
        cols_to_analyze = [col for col in cols_to_analyze if col not in excluded_cols]
    if target_col:
        cols_to_analyze = [col for col in cols_to_analyze if col != target_col]

    for col in cols_to_analyze:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]

        stats["outliers"][col] = {
            "count": len(outliers),
            "percentage": (len(outliers) / len(df)) * 100,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    return processed_df, stats


def analyze_variable_ranges(
    df: pd.DataFrame,
    columns: List[str],
    excluded_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Get the value range and statistics for specified variables.
    Args:
        df: The input DataFrame
        columns: List of column names to analyze
        excluded_cols: List of columns to exclude from analysis (default: None)
        verbose: Whether to print analysis information (default: True)
    Returns:
        Dict[str, Dict]: Dictionary with variable statistics
    """
    if excluded_cols:
        columns = [col for col in columns if col not in excluded_cols]

    stats = {}
    for col in columns:
        non_null_values = df[col].dropna()
        total_count = len(df[col])
        non_null_count = len(non_null_values)
        missing_count = total_count - non_null_count

        col_stats = {
            "unique_count": non_null_values.nunique(),
            "total_count": total_count,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_percentage": (missing_count / total_count) * 100,
        }

        # Add min/max for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update(
                {
                    "min": non_null_values.min(),
                    "max": non_null_values.max(),
                }
            )

        stats[col] = col_stats

        if verbose:
            if pd.api.types.is_numeric_dtype(df[col]):
                print(
                    f"Variable '{col}': Min = {col_stats['min']}, Max = {col_stats['max']} "
                    f"(based on {non_null_count:,} non-null values, {missing_count:,} missing)"
                )
            else:
                print(
                    f"Variable '{col}': {col_stats['unique_count']} unique levels "
                    f"(based on {non_null_count:,} non-null values, {missing_count:,} missing)"
                )

    return stats


def handle_missing_target(
    df: pd.DataFrame,
    target_col: str,
    excluded_cols: Optional[List[str]] = None,
    target_type: type = int,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Handle missing values in target variable.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        excluded_cols: List of columns that should not contain target_col
        target_type: Expected type of target variable (default: int)
        verbose: Whether to print processing information

    Returns:
        Tuple of (processed DataFrame, processing statistics)
    """
    # Validate target column is not in excluded columns
    if excluded_cols and target_col in excluded_cols:
        raise ValueError(f"Target column '{target_col}' cannot be in excluded_cols")

    # Initialize statistics
    stats = {
        "original_rows": len(df),
        "missing_count": df[target_col].isna().sum(),
        "removed_rows": 0,
        "removed_percentage": 0.0,
        "remaining_rows": len(df),
    }

    # Remove rows with missing target values
    df_clean = df[df[target_col].notna()].copy()

    # Update statistics
    stats["removed_rows"] = stats["original_rows"] - len(df_clean)
    stats["removed_percentage"] = (stats["removed_rows"] / stats["original_rows"]) * 100
    stats["remaining_rows"] = len(df_clean)

    if verbose:
        print("\n=== Target Variable Cleaning ===")
        print(f"Original rows: {stats['original_rows']}")
        print(f"Rows with missing target: {stats['missing_count']}")
        print(
            f"Removed {stats['removed_rows']} rows "
            f"({stats['removed_percentage']:.2f}%)"
        )
        print(f"Remaining rows: {stats['remaining_rows']}")

    return df_clean, stats


def analyze_multiple_records(
    df: pd.DataFrame, id_col: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze entities with multiple records and create count features.

    Args:
        df: Input DataFrame
        id_col: Column name containing entity IDs
        verbose: Whether to print analysis information

    Returns:
        DataFrame with added record count features
    """
    df_new = df.copy()

    # Count records per entity
    record_counts = df[id_col].value_counts()

    # Create count feature
    df_new[f"{id_col}_record_count"] = df_new[id_col].map(record_counts)

    # Create flag for multiple records
    df_new[f"has_multiple_{id_col}_records"] = (
        df_new[f"{id_col}_record_count"] > 1
    ).astype(int)

    if verbose:
        print(f"\n=== Multiple Records Analysis for {id_col} ===")
        print(f"Total unique entities: {len(record_counts)}")
        print(f"Entities with multiple records: {sum(record_counts > 1)}")
        print("\nRecord count distribution:")
        print(record_counts.value_counts().sort_index())

    return df_new


def check_record_similarity(
    df: pd.DataFrame,
    id_col: str,
    entity_id: str,
    exclude_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict:
    """
    Check feature similarity across multiple records from the same entity.

    Args:
        df: Input DataFrame
        id_col: Column name containing entity IDs
        entity_id: Specific entity ID to analyze
        exclude_cols: Columns to exclude from similarity check
        verbose: Whether to print analysis information

    Returns:
        Dictionary containing similarity analysis results
    """
    # Get all records for this entity
    entity_records = df[df[id_col] == entity_id].copy()

    if len(entity_records) <= 1:
        if verbose:
            print(f"Entity {entity_id} has only one record.")
        return {"single_record": True}

    if verbose:
        print(f"\nAnalyzing {len(entity_records)} records for entity {entity_id}:")

    results = {"single_record": False, "features": {}}

    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = [id_col]
    else:
        exclude_cols = list(set(exclude_cols + [id_col]))

    # Analyze each feature
    for col in entity_records.columns:
        if col in exclude_cols:
            continue

        values = entity_records[col].unique()
        is_identical = len(values) == 1

        results["features"][col] = {
            "identical": is_identical,
            "unique_values": values.tolist(),
        }

        if verbose:
            if is_identical:
                print(f"- {col}: Same value across all records: {values[0]}")
            else:
                print(f"- {col}: Different values: {values}")

    return results


def analyze_record_independence(
    df: pd.DataFrame,
    id_col: str,
    exclude_cols: Optional[List[str]] = None,
    sample_size: int = 3,
    verbose: bool = True,
) -> Dict:
    """
    Analyze record independence for entities with multiple records.

    Args:
        df: Input DataFrame
        id_col: Column name containing entity IDs
        exclude_cols: Columns to exclude from independence check
        sample_size: Number of sample entities to analyze in detail
        verbose: Whether to print analysis information

    Returns:
        Dictionary containing independence analysis results
    """
    # Get entities with multiple records
    multiple_records = df[id_col].value_counts()
    multiple_records = multiple_records[multiple_records > 1]

    if verbose:
        print(f"\n=== Record Independence Analysis for {id_col} ===")
        print(f"Analyzing {len(multiple_records)} entities with multiple records...")

    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = [id_col]
    else:
        exclude_cols = list(set(exclude_cols + [id_col]))

    # Check for duplicate feature sets
    duplicates = check_duplication(df, excluded_cols=exclude_cols, verbose=False)

    results = {
        "entities_with_multiple_records": len(multiple_records),
        "has_duplicates": duplicates is not None,
        "sample_analyses": {},
    }

    if duplicates is not None:
        results["duplicate_sets"] = len(duplicates) // 2
        if verbose:
            print("\nWARNING: Found identical feature sets!")
            print(f"Number of duplicate feature sets: {results['duplicate_sets']}")

    # Analyze sample entities
    sample_entities = multiple_records.head(sample_size).index
    for entity_id in sample_entities:
        results["sample_analyses"][entity_id] = check_record_similarity(
            df, id_col, entity_id, exclude_cols, verbose
        )

    return results


def deduplicate_records(
    df: pd.DataFrame,
    id_col: str,
    strategy: str = "first",
    excluded_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Deduplicate records based on specified strategy.

    Args:
        df: Input DataFrame
        id_col: Column name containing entity IDs
        strategy: Deduplication strategy ('first', 'last', or 'random')
        exclude_cols: Columns to exclude from duplication check
        verbose: Whether to print deduplication information

    Returns:
        Tuple of (deduplicated DataFrame, deduplication statistics)
    """
    if strategy not in ["first", "last", "random"]:
        raise ValueError("Strategy must be one of: 'first', 'last', 'random'")

    # Analyze duplicates before deduplication
    record_counts = df[id_col].value_counts()
    duplicated_entities = record_counts[record_counts > 1]

    stats = {
        "original_rows": len(df),
        "unique_entities": len(record_counts),
        "entities_with_duplicates": len(duplicated_entities),
        "total_duplicate_rows": sum(duplicated_entities) - len(duplicated_entities),
    }

    # Perform deduplication
    if excluded_cols:
        cols_for_dup_check = [col for col in df.columns if col not in excluded_cols]
    else:
        cols_for_dup_check = df.columns.tolist()

    if strategy == "random":
        df_dedup = df.sample(frac=1).drop_duplicates(
            subset=cols_for_dup_check, keep="first"
        )
    else:
        df_dedup = df.drop_duplicates(subset=cols_for_dup_check, keep=strategy)

    stats["remaining_rows"] = len(df_dedup)
    stats["removed_rows"] = stats["original_rows"] - stats["remaining_rows"]

    if verbose:
        print("\n=== Deduplication Summary ===")
        print(f"Original rows: {stats['original_rows']}")
        print(f"Unique entities: {stats['unique_entities']}")
        print(f"Entities with duplicates: {stats['entities_with_duplicates']}")
        print(f"Total duplicate rows removed: {stats['removed_rows']}")
        print(f"Remaining rows: {stats['remaining_rows']}")

    return df_dedup, stats


def exclude_columns(
    df: pd.DataFrame, excluded_cols: Optional[List[str]] = None, verbose: bool = True
) -> pd.DataFrame:
    """
    Exclude specified columns from DataFrame with validation.

    Args:
        df: Input DataFrame
        excluded_cols: List of columns to exclude
        verbose: Whether to print warning messages

    Returns:
        DataFrame with specified columns excluded
    """
    if excluded_cols is None or len(excluded_cols) == 0:
        return df.copy()

    # Validate columns
    valid_cols = []
    invalid_cols = []

    for col in excluded_cols:
        if col in df.columns:
            valid_cols.append(col)
        else:
            invalid_cols.append(col)

    if len(invalid_cols) > 0 and verbose:
        print("\n=== Column Exclusion Warning ===")
        print("The following columns were not found in the DataFrame:")
        for col in invalid_cols:
            print(f"- {col}")

    if len(valid_cols) == 0:
        if verbose:
            print("No valid columns to exclude. Returning original DataFrame.")
        return df.copy()

    # Exclude valid columns
    df_processed = df.drop(columns=valid_cols)

    if verbose:
        print(f"\nExcluded {len(valid_cols)} columns:")
        for col in valid_cols:
            print(f"- {col}")

    return df_processed


def clean_data(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    excluded_cols: Optional[List[str]] = None,
    dedup_strategy: str = "first",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive data cleaning function that handles deduplication, missing targets,
    and column exclusion.

    Args:
        df: Input DataFrame
        id_col: Column name containing entity IDs for deduplication
        target_col: Name of target column
        excluded_cols: List of columns to exclude
        dedup_strategy: Deduplication strategy ('first', 'last', or 'random')
        verbose: Whether to print processing information

    Returns:
        Tuple of (cleaned DataFrame, cleaning statistics)
    """
    stats = {"original_shape": df.shape, "steps": {}}

    # Step 1: Check duplications
    if verbose:
        print("\nStep 1: Checking duplications...")

    duplicates = check_duplication(df, excluded_cols=excluded_cols, verbose=verbose)
    stats["steps"]["duplication_check"] = {
        "has_duplicates": duplicates is not None,
        "duplicate_rows": len(duplicates) if duplicates is not None else 0,
    }

    # Step 2: Deduplicate records
    if verbose:
        print("\nStep 2: Deduplicating records...")

    df_dedup, dedup_stats = deduplicate_records(
        df,
        id_col=id_col,
        strategy=dedup_strategy,
        excluded_cols=excluded_cols,
        verbose=verbose,
    )
    stats["steps"]["deduplication"] = dedup_stats

    # Step 3: Handle missing target values
    if verbose:
        print("\nStep 3: Handling missing target values...")

    df_clean, target_stats = handle_missing_target(
        df_dedup, target_col=target_col, excluded_cols=excluded_cols, verbose=verbose
    )
    stats["steps"]["target_cleaning"] = target_stats

    # Step 4: Exclude specified columns
    if verbose:
        print("\nStep 4: Excluding specified columns...")

    df_final = exclude_columns(df_clean, excluded_cols=excluded_cols, verbose=verbose)
    stats["steps"]["column_exclusion"] = {
        "columns_excluded": excluded_cols if excluded_cols else [],
        "final_columns": list(df_final.columns),
    }

    # Final statistics
    stats["final_shape"] = df_final.shape
    stats["total_rows_removed"] = stats["original_shape"][0] - stats["final_shape"][0]
    stats["total_columns_removed"] = (
        stats["original_shape"][1] - stats["final_shape"][1]
    )

    if verbose:
        print("\n=== Final Cleaning Summary ===")
        print(f"Original shape: {stats['original_shape']}")
        print(f"Final shape: {stats['final_shape']}")
        print(f"Total rows removed: {stats['total_rows_removed']}")
        print(f"Total columns removed: {stats['total_columns_removed']}")

    return df_final, stats
