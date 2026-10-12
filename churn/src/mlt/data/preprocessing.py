from logging import Logger

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

numerics = ["tenure", "MonthlyCharges"]
ordinal = ["Contract"]
categorical = [
    "InternetService",
    "PaymentMethod",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "SeniorCitizen",
    "OnlineSecurity",
    "is_new_customer",
    "high_spender",
]

# vielleucht features auslagern


def get_data(logger: str, INTERIM_DATA_DIR: str) -> list:
    logger.info("loading preprocessed data")
    data = pd.read_csv(str(INTERIM_DATA_DIR) + "/telco_clean.csv")

    y = data["Churn"]
    X = data[categorical + numerics + ordinal]

    return X, y


def preprocessing_pipeline(logger: Logger) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ordinal_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numerics),
            ("cat", categorical_transformer, categorical),
            ("ord", ordinal_transformer, ordinal),
        ]
    )

    logger.info("preprocessor ready to be deployed")

    return preprocessor
