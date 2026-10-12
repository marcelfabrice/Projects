# Customer Churn Prediction

This project predicts customer churn on the Telco Customer Churn dataset.

## Setup

Install and run the package with:

```bash
uv run mlt
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

For development, install the pre-commit hooks:

```bash
pre-commit install
```

## Pipeline

Run the full DVC pipeline:

```bash
dvc repro -f
```

Start the MLflow UI to inspect model runs:

```bash
mlflow ui --port 8080
```

## Project Contents

- Exploratory notebooks for data and model experiments
- Source files for configuration, data cleaning, preprocessing, model fitting, metrics, and hyperparameter optimization
- DVC and MLflow tracking data
- Log files
- Raw, interim, and processed data
- Parameters in `params.yaml`


## Data Cleaning And Preprocessing

The cleaning step formats the raw Telco data for model preprocessing.

Preprocessing includes:

- Missing-value imputation
- One-hot encoding for categorical features
- Ordinal encoding for contract data
- Standard scaling for numeric features

## Baseline

The baseline model is logistic regression.

Baseline parameters:

- `max_iter`: 1500
- `class_weight`: `balanced`

Class weights are balanced because the churn target is imbalanced.
