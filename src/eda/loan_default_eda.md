# Loan Default Prediction - Exploratory Data Analysis (EDA)

## Introduction

This report summarizes the exploratory data analysis conducted on the loan default prediction dataset. The goal is to understand the dataset's structure, identify key variables, and analyze the target variable, `bad_flag`, which indicates whether a loan is considered "bad."

## Data Inspection

###  Data Loading

The dataset was loaded from the following files:

- **Training Data**: `data/training_loan_data.csv`
- **Data Dictionary**: `data/dict_data.json`

### Column Verification

- **Dataset Shape**: (199121, 23)
- **Number of Columns in Data Dictionary**: 23
- **Number of Columns in Dataset**: 23

### Columns in DataFrame

The following columns were verified against the data dictionary:

| Column Name                 | Status | Description                                                                                                                                                                                           | Variable Type | Missingness     |
| --------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | --------------- |
| annual_inc                  | ✓      | The annual income provided by the borrower during application.                                                                                                                                        | Continuous    | 9664 (4.85%)    |
| bc_util                     | ✓      | Ratio of total current balance to high credit/credit limit for all bankcard accounts.                                                                                                                 | Continuous    | 18788 (9.44%)   |
| desc                        | ✓      | Loan description provided by the borrower.                                                                                                                                                            | Categorical   | 117117 (58.82%) |
| dti                         | ✓      | A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage and the requested loan, divided by the borrower's self-reported monthly income. | Continuous    | 9664 (4.85%)    |
| emp_length                  | ✓      | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.                                                                     | Categorical   | 0 (0%)          |
| home_ownership              | ✓      | The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.                                                                                   | Categorical   | 9664 (4.85%)    |
| id                          | ✓      | A unique assigned ID for the loan listing.                                                                                                                                                            | Categorical   | 0 (0%)          |
| inq_last_6mths              | ✓      | The number of inquiries by creditors during the past 6 months.                                                                                                                                        | Continuous    | 9664 (4.85%)    |
| int_rate                    | ✓      | Interest Rate on the loan.                                                                                                                                                                            | Categorical   | 9664 (4.85%)    |
| loan_amnt                   | ✓      | The listed amount of the loan applied for by the borrower.                                                                                                                                            | Continuous    | 0 (0%)          |
| member_id                   | ✓      | A unique assigned Id for the borrower member.                                                                                                                                                         | Categorical   | 0 (0%)          |
| mths_since_last_major_derog | ✓      | Months since most recent 90-day or worse rating.                                                                                                                                                      | Continuous    | 166372 (83.55%) |
| mths_since_recent_inq       | ✓      | Months since most recent inquiry.                                                                                                                                                                     | Continuous    | 37649 (18.91%)  |
| percent_bc_gt_75            | ✓      | Percentage of all bankcard accounts > 75% of limit.                                                                                                                                                   | Continuous    | 18702 (9.39%)   |
| purpose                     | ✓      | A category provided by the borrower for the loan request.                                                                                                                                             | Categorical   | 9664 (4.85%)    |
| revol_util                  | ✓      | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.                                                                            | Categorical   | 9791 (4.92%)    |
| term                        | ✓      | The number of payments on the loan. Values are in months and can be either 36 or 60.                                                                                                                  | Categorical   | 0 (0%)          |
| tot_cur_bal                 | ✓      | Total current balance of all accounts.                                                                                                                                                                | Continuous    | 37405 (18.79%)  |
| tot_hi_cred_lim             | ✓      | Total high credit/credit limit.                                                                                                                                                                       | Continuous    | 17159 (8.62%)   |
| total_bc_limit              | ✓      | Total bankcard high credit/credit limit.                                                                                                                                                              | Continuous    | 17159 (8.62%)   |
| application_approved_flag   | ✓      | Indicates if the loan application is approved or not.                                                                                                                                                 | Continuous    | 0 (0%)          |
| internal_score              | ✓      | A third party vendor's risk score generated when the application is made.                                                                                                                             | Continuous    | 0 (0%)          |
| bad_flag                    | ✓      | Target variable, indicates if the loan is eventually bad or not.                                                                                                                                      | Continuous    | 9664 (4.85%)    |

### Target Variable Analysis

#### Distribution of `bad_flag`

- Class distribution:
  - Class 0: 176329 (93.07%)
  - Class 1: 13128 (6.93%)

### Conclusion

This exploratory data analysis provides a foundational understanding of the loan default prediction dataset.

1. **Dataset Size**: The dataset is of moderate size, containing 199,121 rows and 23 columns, which is suitable for various analytical and modeling techniques.

2. **Label Imbalance**: The dataset exhibits some label imbalance, with 93.07% of the instances classified as Class 0 (not bad loans) and 6.93% as Class 1 (bad loans). While this imbalance is present, it is not excessively skewed, allowing for potential modeling strategies to address the imbalance effectively.

3. **Missingness**: Overall, the dataset has an acceptable level of missing values. However, two variables, `mths_since_last_major_derog` and `desc`, have significant missingness, which may require special attention during the modeling process to ensure robust predictions.
