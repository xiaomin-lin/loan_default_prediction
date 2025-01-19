"""Feature engineer for loan-specific features."""

import pandas as pd


class LoanFeatureEngineer:
    """Engineer raw loan features into engineered features."""

    def __init__(self):
        """Initialize the engineer."""
        pass

    def fit(self, df: pd.DataFrame) -> "LoanFeatureEngineer":
        """Fit is a no-op for this engineer."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform loan features."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        required_cols = [
            "tot_cur_bal",
            "annual_inc",
            "tot_hi_cred_lim",
            "bc_util",
            "revol_util",
            "loan_amnt",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(
                f"Warning: Missing columns {missing_cols} will skip related feature engineering"
            )

        df_new = df.copy()

        # Total balance to income ratio
        if all(col in df.columns for col in ["tot_cur_bal", "annual_inc"]):
            df_new["total_balance_to_income"] = df["tot_cur_bal"] / (
                df["annual_inc"] + 1e-8
            )

        # Overall credit utilization
        if all(col in df.columns for col in ["tot_cur_bal", "tot_hi_cred_lim"]):
            df_new["total_utilization"] = df["tot_cur_bal"] / (
                df["tot_hi_cred_lim"] + 1e-8
            )
            df_new["high_total_utilization"] = (
                df_new["total_utilization"] > 0.8
            ).astype(int)

        # Bankcard utilization
        if "bc_util" in df.columns:
            df_new["bankcard_utilization"] = df["bc_util"] / 100
            df_new["high_bankcard_utilization"] = (df["bc_util"] > 75).astype(int)

        # Revolving utilization
        if "revol_util" in df.columns:
            df_new["revol_util_ratio"] = df["revol_util"] / 100
            df_new["high_revol_utilization"] = (df["revol_util"] > 80).astype(int)

        # Loan amount features
        if all(col in df.columns for col in ["loan_amnt", "annual_inc"]):
            df_new["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-8)

        return df_new
