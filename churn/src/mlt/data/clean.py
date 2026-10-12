import pandas as pd

from mlt.config import INTERIM_DATA_DIR, RAW_DATA_DIR, get_logger


def clean_data() -> None:
    logger = get_logger(__name__)

    """
    cleans data:
        converts string to floats
        replaces definite values for yes/no
        strips special characters or unnecessary prefixes/suffixes
        drops unnecessary features
        creates new features from numerical data
        encodes target feature
        deletes outlier (trash data)
        saves cleaned dataframe to file
    """

    logger.info("start data cleaning")

    df = pd.read_csv(RAW_DATA_DIR / "telco_data.csv")

    # Dollar entfernen und in float konvertieren
    df["MonthlyCharges"] = df["MonthlyCharges"].str.lstrip("$").astype(float)

    # gender bezeichnungen einheitlich
    df["gender"] = df["gender"].str.lower().str.strip()
    df["gender"] = df["gender"].replace(
        {"m": "male", "male": "male", "f": "female", "female": "female"}
    )

    # Partner beizeichnungen einheitlich
    df["Partner"] = df["Partner"].str.lower().str.strip()
    df["Partner"] = df["Partner"].replace({"y": "yes", "n": "no"})

    # Tenure month enfernen und in float konvertieren
    df["tenure"] = df["tenure"].str.strip(" months").astype(float)

    # Dollar hinten enfernen
    df["TotalCharges"] = df["TotalCharges"].str.strip(" USD").astype(float)

    # Senior citzien
    df["SeniorCitizen"] = df["SeniorCitizen"].str.lower().str.strip()
    df["SeniorCitizen"] = df["SeniorCitizen"].replace(
        {"y": "yes", "n": "no", "0": "no", "1": "yes"}
    )

    # Dependants yes/no
    df["Dependents"] = df["Dependents"].str.lower().str.strip()
    df["Dependents"] = df["Dependents"].replace({"y": "yes", "n": "no"})

    # PhoneService yes/no
    df["PhoneService"] = df["PhoneService"].str.lower().str.strip()
    df["PhoneService"] = df["PhoneService"].replace({"y": "yes", "n": "no"})

    logger.info("datatypes and formalities corrected")
    logger.info("encoding and dropping data...")

    # neue features engineered
    df["is_new_customer"] = (df["tenure"] < 12).astype(int)
    df["high_spender"] = (df["MonthlyCharges"] > 80).astype(int)

    df = df.drop(
        [
            "StreamingMovies",
            "StreamingTV",
            "MultipleLines",
            "PhoneService",
            "TotalCharges",
            "Partner",
            "gender",
            "customer_hash",
            "customerID",
        ],
        axis=1,
    )

    # Ausreißer müll entfernen
    df = df[df["tenure"] < 80]

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logger.info("data finished ready to train")

    df.to_csv(INTERIM_DATA_DIR / "telco_clean.csv", index=False)


if __name__ == "__main__":
    clean_data()
