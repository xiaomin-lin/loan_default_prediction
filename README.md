# Loan Default Prediction Project

## Description

This project aims to predict loan defaults using various machine learning techniques. The `src` directory contains the main application code, while the `tests` directory includes unit tests for the application.

## Project Structure

This project is organized as follows:

```
project_root/
│
├── data/                       # Data directory
│   └── training_loan_data.csv  # Training data
│   └── data_dictionary.json    # Data dictionary
│   └── test_loan_data.csv      # Test data
│
├── notebooks/                  # Notebooks directory
│
├── reports/                    # Reports directory
│  └── eda/
│      └── plots/                 # EDA plots
│      └── eda_report.md          # EDA markdown report
│
├── src/                          # Source code directory
│   ├── eda/                     # Exploratory Data Analysis
│   │   ├── data_inspection.py   # Initial data inspection and loading
│   │   ├── data_quality.py      # Data quality checks and cleaning
│   │   ├── eda.py              # Main EDA analysis and visualization
│   │   └── __init__.py
│   ├── utils/                   # Utility functions
│   │   ├── data_utils.py       # Common data processing utilities
│   │   └── __init__.py
│   ├── features/               # Feature engineering (planned)
│   │   └── __init__.py
│   ├── models/                 # Model training and evaluation (planned)
│   │   └── __init__.py
│   ├── data/                   # Data processing scripts (planned)
│   │   └── __init__.py
│   ├── __init__.py
│   └── __main__.py            # Main entry point
│
├── tests/                      # Test directory
│   ├── test_data_inspection.py
│   ├── test_data_quality.py
│   └── __init__.py
│
├── requirements.txt           # Project dependencies
├── setup.py                  # Package setup configuration
└── README.md                 # Project documentation
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/xiaomin-lin/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the project and all dependencies:
   ```bash
   pip install -e .  # This will also install everything from requirements.txt
   ```

## Usage

### Data Inspection and Quality

To run the initial data inspection:

```bash
python -m src.eda.data_inspection
```

To perform data quality checks:

```bash
python -m src.eda.data_quality
```

### Exploratory Data Analysis

To generate EDA visualizations and analysis:

```bash
python -m src.eda.eda
```

## Project Components

### EDA Module

- `data_inspection.py`: Handles data loading and initial inspection
- `data_quality.py`: Performs data quality checks and cleaning
- `eda.py`: Generates visualizations and statistical analysis

### Utils Module

- `data_utils.py`: Contains common utility functions for data processing

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author

Xiaomin Lin
