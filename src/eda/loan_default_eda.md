# Loan Default Prediction - Exploratory Data Analysis (EDA)

## Table of Contents

1. [Introduction](#introduction)
2. [Data Inspection and Quality Control](#data-inspection-and-quality-control)
   - [Data Loading](#data-loading)
   - [Column Verification](#column-verification)
   - [Preliminary Data Transformation](#preliminary-data-transformation)
   - [Outlier Analysis](#outlier-analysis)
   - [Data Profile](#data-profile)
   - [Target Variable Analysis](#target-variable-analysis)
   - [Findings](#findings)
3. [Data Exploratory Analysis](#data-exploratory-analysis)
   - [Univariate Analysis](#univariate-analysis)
   - [Variable Importance Analysis](#variable-importance-analysis)
   - [Inter-predictor Correlation Analysis](#inter-predictor-correlation-analysis)
   - [Feature Correlations with Target](#feature-correlations-with-target)
4. [Appendix](#appendix)
   - [Univariate Analysis Plots](#univariate-analysis-plots)
   - [Correlation Heatmap](#correlation-heatmap)
   - [Variable Importance Plot](#variable-importance-plot)
   - [Feature Correlations Plot](#feature-correlations-plot)

## Introduction

This report presents a comprehensive exploratory data analysis (EDA) of a loan default prediction dataset. In the context of consumer lending, understanding and predicting loan defaults is crucial for risk management and portfolio optimization. Through this analysis, we aim to uncover the underlying patterns, relationships, and key predictors that influence loan default probability.

Our investigation follows a structured approach, beginning with rigorous data quality assessment and preliminary transformations. We then delve into detailed analyses of variable distributions, outlier patterns, and missing value mechanisms. Particular attention is paid to the target variable, `bad_flag`, which indicates loan default status, and its relationships with various predictors through correlation and information value analyses.

The insights derived from this exploration will serve multiple purposes: guiding feature engineering decisions, informing modeling choices, and establishing baseline expectations for predictive performance. By understanding the nuances of our dataset, we lay the foundation for developing a robust and interpretable loan default prediction model that can effectively distinguish between high and low-risk borrowers.

## Data Inspection and Quality Control

### Data Loading

The dataset was loaded from the following files:

- **Training Data**: `data/training_loan_data.csv`
- **Data Dictionary**: `data/dict_data.json`

### Column Verification

- **Dataset Shape**: (199121, 23)
- **Number of Columns in Data Dictionary**: 23
- **Number of Columns in Dataset**: 23

### Preliminary Data Transformation

Before conducting detailed analysis, several variables required preliminary transformation to ensure proper data types and formats:

1. **ID Variables**:

   - `id` and `member_id`: Converted to string type to preserve their full precision and prevent numerical interpretation

2. **Numeric Variables**:

   - `loan_amnt`: Converted to integer type
   - `int_rate`: Removed '%' symbol and converted to float (e.g., "15.25%" → 0.1525)
   - `revol_util`: Removed '%' symbol and converted to float (e.g., "50.25%" → 0.5025)

3. **Categorical Variables**:
   - `term`: Converted to string type to preserve categorical nature
   - `emp_length`: Special handling:
     - Converted text to numeric values (e.g., "5 years" → 5)
     - Null values preserved
     - "< 1 year" converted to 0
     - "10+ years" converted to 10

These transformations ensure that:

- Variables are in their appropriate data types for analysis
- Percentage values are in decimal format for mathematical operations
- Categorical variables are properly encoded
- Special text patterns are consistently handled

### Outlier Analysis

The dataset contains several variables with notable outliers. We used the Interquartile Range (IQR) method [Tukey, J. W. (1977). Exploratory Data Analysis.] to identify outliers in continuous variables. This method defines outliers as values that fall outside the range:

- Lower bound: Q1 - 1.5 × IQR
- Upper bound: Q3 + 1.5 × IQR

where Q1 is the first quartile, Q3 is the third quartile, and IQR = Q3 - Q1.

Key findings from outlier analysis:

1. **High Impact Variables**:

   - `inq_last_6mths`: 14,777 outliers (7.80%)
   - `tot_hi_cred_lim`: 10,219 outliers (5.39%)
   - `annual_inc`: 7,550 outliers (3.99%)

2. **Moderate Impact Variables**:
   - `tot_cur_bal`: 4,778 outliers (2.52%)
   - `bc_util`: 7 outliers (0.00%)
   - `mths_since_last_major_derog`: 43 outliers (0.02%)

For categorical variables, while rare categories could be considered outliers, this approach was not applied as it may not be meaningful for this lending context where rare categories (e.g., specific loan purposes) could still be valid and important predictors.

### Data Profile

The following table were verified against the data dictionary, as well as incorporting the some of results from the above:

| Column Name                 | Description                                                                                                                                                                                           | Variable Type | Value Range/Level           | Missingness     | Outlier       |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | --------------------------- | --------------- | ------------- |
| annual_inc                  | The annual income provided by the borrower during application.                                                                                                                                        | Continuous    | 4,800 - 7,141,778           | 9664 (4.85%)    | 7550 (3.99%)  |
| bc_util                     | Ratio of total current balance to high credit/credit limit for all bankcard accounts.                                                                                                                 | Continuous    | 0 - 339.6                   | 18788 (9.44%)   | 7 (0.00%)     |
| desc                        | Loan description provided by the borrower.                                                                                                                                                            | Categorical   | -                           | 117117 (58.82%) | -             |
| dti                         | A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested loan, divided by the borrower's self-reported monthly income. | Continuous    | 0 - 34.99                   | 9664 (4.85%)    | -             |
| emp_length                  | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.                                                                     | Continuous    | 1 - 10                      | 0 (0%)          | -             |
| home_ownership              | The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.                                                                                   | Categorical   | 5 levels                    | 9664 (4.85%)    | -             |
| id                          | A unique assigned ID for the loan listing.                                                                                                                                                            | Categorical   | -                           | 0 (0%)          | -             |
| inq_last_6mths              | The number of inquiries by creditors during the past 6 months.                                                                                                                                        | Continuous    | 0 - 8                       | 9664 (4.85%)    | 14777 (7.80%) |
| int_rate                    | Interest Rate on the loan.                                                                                                                                                                            | Continuous    | 0.06 - 0.2606 (6% - 26.06%) | 9664 (4.85%)    | -             |
| loan_amnt                   | The listed amount of the loan applied for by the borrower.                                                                                                                                            | Continuous    | 1,000 - 35,000              | 0 (0%)          | -             |
| member_id                   | A unique assigned Id for the borrower member.                                                                                                                                                         | Categorical   | -                           | 0 (0%)          | -             |
| mths_since_last_major_derog | Months since most recent 90-day or worse rating.                                                                                                                                                      | Continuous    | 0 - 165                     | 166372 (83.55%) | 43 (0.02%)    |
| mths_since_recent_inq       | Months since most recent inquiry.                                                                                                                                                                     | Continuous    | 0 - 24                      | 37649 (18.91%)  | -             |
| percent_bc_gt_75            | Percentage of all bankcard accounts > 75% of limit.                                                                                                                                                   | Continuous    | 0 - 100                     | 18702 (9.39%)   | -             |
| purpose                     | A category provided by the borrower for the loan request.                                                                                                                                             | Categorical   | 13 levels                   | 9664 (4.85%)    | -             |
| revol_util                  | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.                                                                            | Continuous    | 0 - 1.404 (0% - 140.4%)     | 9791 (4.92%)    | -             |
| term                        | The number of payments on the loan. Values are in months and can be either 36 or 60.                                                                                                                  | Categorical   | 2 levels (36, 60 months)    | 0 (0%)          | -             |
| tot_cur_bal                 | Total current balance of all accounts.                                                                                                                                                                | Continuous    | 0 - 8,000,078               | 37405 (18.79%)  | 4778 (2.52%)  |
| tot_hi_cred_lim             | Total high credit/credit limit.                                                                                                                                                                       | Continuous    | 0 - 520,643.30              | 17159 (8.62%)   | 10219 (5.39%) |
| total_bc_limit              | Total bankcard high credit/credit limit.                                                                                                                                                              | Continuous    | 0 - 522,210                 | 17159 (8.62%)   | -             |
| application_approved_flag   | Indicates if the loan application is approved or not.                                                                                                                                                 | Boolean       | 1                           | 0 (0%)          | -             |
| internal_score              | A third party vendor's risk score generated when the application is made.                                                                                                                             | Continuous    | 14 - 456                    | 0 (0%)          | -             |
| bad_flag                    | Target variable, indicates if the loan is eventually bad or not.                                                                                                                                      | Boolean       | 0 - 1                       | 9664 (4.85%)    | -             |

### Target Variable Analysis

#### Distribution of `bad_flag`

- Class distribution:
  - Class 0: 176329 (93.07%)
  - Class 1: 13128 (6.93%)

### Findings

Our preliminary inspection of the dataset as well as its quality shows that overall, the dataset is of good amount and good quality. Specifically:

1. **Dataset Size**: The dataset is of moderate size, containing 199,121 rows and 23 columns, which is suitable for various analytical and modeling techniques. From a hardware perspective, this dataset is not too large to be processed on a single machine with CPU.

2. **Label Imbalance**: The dataset exhibits some label imbalance, with 93.07% of the instances classified as Class 0 (not bad loans) and 6.93% as Class 1 (bad loans). While this imbalance is present, it is not excessively skewed, allowing for potential modeling strategies to address the imbalance effectively.

3. **Missingness**: Overall, the dataset has an acceptable level of missing values. However, two variables, `mths_since_last_major_derog` and `desc`, have significant missingness, which may require special attention during the modeling process to ensure robust predictions.

4. **Outliers**: The dataset has outliers in several variables, but: 1. they are not extreme, and 2. we have sufficient data to train the model. Therefore, the outliers may not affect the model's performance too much.

5. **Redundancy**: The variable `application_approved_flag` is a redundant variable, as it has only one value, which is 1. Therefore, it can be removed from the dataset from the modeling perspective.

6. **Potentiality**: The variable `desc` contains a lot of text data, which may be useful for the modeling process. However, it is also a very sparse variable, with only 117,117 non-null values out of 199,121 rows. Anyway, it worths some investigation of how to leverage it in the modeling process.

## Data Exploratory Analysis

This section aims to explore the dataset to uncover key patterns in the loan data. We conduct univariate analysis, variable importance analysis, and predictors inter-correlations and their correlations with the target variable.

### Univariate Analysis

All results can be found in [Appendix/Univariate Analysis Plots](#univariate-analysis-plots). Our findings are as follows:

1. **Annual Income**

   - Loan amounts exhibit a multi-modal distribution with major peaks at $10,000, $15,000, and $20,000, suggesting standardized loan products
   - Interest rates range from 6% to 26%, with a right-skewed distribution centered around 13%
   - 36-month terms are significantly more popular than 60-month terms
   - Debt consolidation is the dominant loan purpose, followed by credit card refinancing

2. **Income and Debt Profiles**:

   - Annual incomes show extreme right skew, with median around $65,000 and outliers up to $7M
   - DTI ratios concentrate between 10-25%, with higher ratios showing increased default risk
   - Total current balance and credit limits show large variations, indicating diverse borrower profiles
   - Higher income borrowers tend to have lower default rates

3. **Credit Utilization Patterns**:

   - Both revolving and bankcard utilization show bimodal distributions
   - Many borrowers cluster around either low (<20%) or high (>80%) utilization
   - About 30% of borrowers have at least one bankcard above 75% utilization
   - Higher utilization rates strongly correlate with increased default probability

4. **Credit History Indicators**:

   - Recent inquiries are typically low (0-2), with each additional inquiry associated with higher default risk
   - Employment length shows U-shaped distribution, peaking at both extremes (1 and 10+ years)
   - Major derogatory marks are rare but highly predictive of default risk
   - Internal credit scores show normal distribution, strongly correlating with default probability

5. **Borrower Demographics**:
   - Home ownership is dominated by mortgages (50%) and rentals (40%)
   - Mortgage holders show lower default rates compared to renters
   - Property owners (both mortgage and outright) tend to have higher loan amounts
   - Rental status correlates with higher interest rates and default risk

### Variable Importance Analysis

We employed Information Value (IV) and Weight of Evidence (WOE) analysis to assess feature importance. This methodology is widely adopted in credit risk modeling [Anderson, R. (2007). The Credit Scoring Toolkit; Siddiqi, N. (2006). Credit Risk Scorecards.] for several key reasons:

1. **Universal Applicability**: Works effectively for both continuous and categorical variables without requiring transformation or encoding
2. **Interpretability**: Provides clear, interpretable measures of predictive power
3. **Non-linear Relationships**: Captures non-linear relationships between features and the target variable
4. **Industry Standard**: Recognized as a standard approach in credit scoring and risk assessment

Typically, IV values are interpreted as:

- < 0.02: Unpredictive
- 0.02 to 0.1: Weak
- 0.1 to 0.3: Medium
- \> 0.3: Strong

The Information Value (IV) analysis in [Appendix/Variable Importance Plot](#variable-importance-plot) reveals the predictive power of features for loan default:

1. **High Importance** (IV > 0.3):

   - `loan_amnt` (IV ≈ 0.5): Strongest predictor
   - `int_rate` (IV ≈ 0.3): Second most important feature

2. **Moderate Importance** (0.05 < IV < 0.1):

   - Application characteristics: `purpose`, `mths_since_recent_inq`, `inq_last_6mths`
   - Financial metrics: `annual_inc`, `total_bc_limit`, `tot_hi_cred_lim`
   - Credit utilization: `bc_util`, `percent_bc_gt_75`, `tot_cur_bal`

3. **Lower Importance** (IV < 0.05):
   - `home_ownership`, `revol_util`, `dti`, `term`
   - `emp_length`, `internal_score`, `mths_since_last_major_derog`

This analysis suggests that loan amount and pricing factors are the strongest predictors of default risk, while employment and credit history metrics show lower predictive power.

### Inter-predictor Correlation Analysis

The inter-predictor correlation analysis in [Appendix/Correlation Heatmap](#correlation-heatmap) reveals several important relationships between continuous variables, as follows:

- **Credit Capacity Metrics**: Strong positive correlations (>0.8) exist between total credit limits, bankcard limits, and current balances, suggesting consistent credit utilization patterns across different credit types
- **Risk Indicators**: Internal score shows strong negative correlations with interest rate (-0.7) and moderate negative correlations with DTI (-0.4) and utilization metrics (-0.5), confirming its value as a risk predictor
- **Utilization Metrics**: Revolving utilization and bankcard utilization show moderate correlation (0.6), indicating some borrowers maintain similar utilization levels across credit types
- **Income Relationships**: Annual income shows weak to moderate positive correlations with credit limits (0.3-0.4) and weak negative correlations with utilization metrics (-0.2), suggesting higher income borrowers maintain lower utilization
- **Default Risk Factors**: The target variable (bad_flag) shows strongest correlations with internal score (-0.3), interest rate (0.25), and utilization metrics (0.2), highlighting key risk factors

### Feature Correlations with Target

The feature correlations with target in [Appendix/Feature Correlations Plot](#feature-correlations-plot) reveals several important relationships between continuous variables, as follows:

- **Strong Correlations**: Interest rate shows the strongest positive correlation (≈0.125), indicating higher rates are associated with increased default risk
- **Moderate Correlations**: Loan purpose (Cramer's V) and recent inquiries (`inq_last_6mths`) show moderate positive correlations (0.05-0.075)
- **Weak Positive Correlations**: Credit utilization metrics (`bc_util`, `revol_util`, `percent_bc_gt_75`) and DTI show weak positive correlations (0.025-0.05)
- **Negative Correlations**: Financial capacity indicators (`mths_since_recent_inq`, `total_bc_limit`, `tot_hi_cred_lim`, `annual_inc`) show weak negative correlations (-0.05-0), suggesting higher financial capacity is associated with lower default risk

Overall, these correlation results do not suggest any strong relationships between the features and the target variable.

## Conclusions

In this exploratory data analysis (EDA), we systematically examined the loan default dataset to uncover key insights and relationships among variables. We performed data quality checks, outlier analysis, and assessed feature importance using Information Value (IV) and Weight of Evidence (WOE) methodologies. Additionally, we analyzed inter-predictor correlations and their relationships with the target variable, `bad_flag`.

### Major Findings:

1. **Data Quality**: This dataset is of good amount and good quality in general. Although missingness, outliers, redundancy, and label imbalance are present, they are not severe enough to affect the predictive model's performance given proper data preprocessing and feature engineering.
2. **Feature Importance**: The loan amount and interest rate emerged as the strongest predictors of default risk, while internal scores and credit utilization metrics also demonstrated significant correlations.
3. **Correlation Insights**: Strong positive correlations were identified among credit capacity metrics, indicating consistent utilization patterns, while risk indicators highlighted the importance of pricing and recent credit behavior.

### Next Steps:

The insights gained from this EDA will inform our feature engineering process, allowing us to prioritize the most impactful variables for predictive modeling. Understanding the relationships between features and the target variable will guide the selection of appropriate modeling techniques and enhance the robustness of our predictive models. By leveraging these findings, we aim to build a more accurate and interpretable model for predicting loan defaults.

## Appendix

### Univariate Analysis Plots

#### Continuous Variables

**Annual Income** (The annual income provided by the borrower during application)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/annual_inc_distribution.png" width="32%">
    <img src="../../reports/eda/plots/annual_inc_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/annual_inc_vs_bad_flag_boxplot.png" width="32%">
</div>

**Bankcard Utilization** (Ratio of total current balance to high credit/credit limit)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/bc_util_distribution.png" width="32%">
    <img src="../../reports/eda/plots/bc_util_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/bc_util_vs_bad_flag_boxplot.png" width="32%">
</div>

**DTI** (Debt-to-Income ratio)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/dti_distribution.png" width="32%">
    <img src="../../reports/eda/plots/dti_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/dti_vs_bad_flag_boxplot.png" width="32%">
</div>

**Employment Length** (Employment length in years)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/emp_length_distribution.png" width="32%">
    <img src="../../reports/eda/plots/emp_length_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/emp_length_vs_bad_flag_boxplot.png" width="32%">
</div>

**Recent Inquiries** (Number of inquiries in last 6 months)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/inq_last_6mths_distribution.png" width="32%">
    <img src="../../reports/eda/plots/inq_last_6mths_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/inq_last_6mths_vs_bad_flag_boxplot.png" width="32%">
</div>

**Interest Rate** (Interest Rate on the loan)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/int_rate_distribution.png" width="32%">
    <img src="../../reports/eda/plots/int_rate_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/int_rate_vs_bad_flag_boxplot.png" width="32%">
</div>

**Loan Amount** (The listed amount of the loan applied for)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/loan_amnt_distribution.png" width="32%">
    <img src="../../reports/eda/plots/loan_amnt_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/loan_amnt_vs_bad_flag_boxplot.png" width="32%">
</div>

**Months Since Last Major Derogatory**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/mths_since_last_major_derog_distribution.png" width="32%">
    <img src="../../reports/eda/plots/mths_since_last_major_derog_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/mths_since_last_major_derog_vs_bad_flag_boxplot.png" width="32%">
</div>

**Months Since Recent Inquiry**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/mths_since_recent_inq_distribution.png" width="32%">
    <img src="../../reports/eda/plots/mths_since_recent_inq_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/mths_since_recent_inq_vs_bad_flag_boxplot.png" width="32%">
</div>

**Percent Bankcard > 75%** (Percentage of all bankcard accounts > 75% of limit)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/percent_bc_gt_75_distribution.png" width="32%">
    <img src="../../reports/eda/plots/percent_bc_gt_75_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/percent_bc_gt_75_vs_bad_flag_boxplot.png" width="32%">
</div>

**Revolving Utilization** (Revolving line utilization rate)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/revol_util_distribution.png" width="32%">
    <img src="../../reports/eda/plots/revol_util_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/revol_util_vs_bad_flag_boxplot.png" width="32%">
</div>

**Total Current Balance** (Total current balance of all accounts)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/tot_cur_bal_distribution.png" width="32%">
    <img src="../../reports/eda/plots/tot_cur_bal_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/tot_cur_bal_vs_bad_flag_boxplot.png" width="32%">
</div>

**Total High Credit Limit**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/tot_hi_cred_lim_distribution.png" width="32%">
    <img src="../../reports/eda/plots/tot_hi_cred_lim_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/tot_hi_cred_lim_vs_bad_flag_boxplot.png" width="32%">
</div>

**Total Bankcard Limit**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/total_bc_limit_distribution.png" width="32%">
    <img src="../../reports/eda/plots/total_bc_limit_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/total_bc_limit_vs_bad_flag_boxplot.png" width="32%">
</div>

**Internal Score** (Third party vendor's risk score)

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/internal_score_distribution.png" width="32%">
    <img src="../../reports/eda/plots/internal_score_boxplot.png" width="32%">
    <img src="../../reports/eda/plots/internal_score_vs_bad_flag_boxplot.png" width="32%">
</div>

#### Categorical Variables

**Home Ownership**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/home_ownership_countplot.png" width="48%">
    <img src="../../reports/eda/plots/home_ownership_distribution_by_bad_flag_countplot.png" width="48%">
</div>

**Loan Purpose**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/purpose_countplot.png" width="48%">
    <img src="../../reports/eda/plots/purpose_distribution_by_bad_flag_countplot.png" width="48%">
</div>

**Term**

<div style="display: flex; justify-content: space-between;">
    <img src="../../reports/eda/plots/term_countplot.png" width="48%">
    <img src="../../reports/eda/plots/term_distribution_by_bad_flag_countplot.png" width="48%">
</div>

### Correlation Heatmap

<div style="display: flex; justify-content: center;">
    <img src="../../reports/eda/plots/correlation_heatmap.png" width="100%">
</div>

### Variable Importance Plot

<div style="display: flex; justify-content: center;">
    <img src="../../reports/eda/plots/feature_importance_iv.png" width="100%">
</div>

### Feature Correlations Plot

<div style="display: flex; justify-content: center;">
    <img src="../../reports/eda/plots/feature_correlations.png" width="100%">
</div>
