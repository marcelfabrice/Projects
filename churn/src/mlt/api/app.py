from contextlib import asynccontextmanager

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException

from mlt.api.config import settings
from mlt.api.schemas import Customer, Prediction
from mlt.config import INTERIM_DATA_DIR, get_logger

logger = get_logger(__name__)

models: dict = {}
CUSTOMER_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "SeniorCitizen",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "OnlineBackup",
    "TechSupport",
    "DeviceProtection",
    "Contract",
]
BACKGROUND_SIZE = 20
EXPLAINER_LINK = "identity"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.knn_model_path.exists():
        raise RuntimeError("model path could not be resolved")

    logger.info("loading model from %s", settings.knn_model_path)

    # model von datei nach training laden
    models["knn_churn"] = joblib.load(settings.knn_model_path)
    background = pd.read_csv(INTERIM_DATA_DIR / "telco_clean.csv")
    models["shap_background"] = background[CUSTOMER_FEATURES].sample(
        n=min(BACKGROUND_SIZE, len(background)),
        random_state=42,
    )
    models["shap_explainer"] = shap.KernelExplainer(
        predict_probabilities,
        models["shap_background"],
        link=EXPLAINER_LINK,
    )

    logger.info("model loaded")
    yield
    models.clear()


app = FastAPI(
    title="KNN-Telco Churn Prediction",
    description="Predict wether a customer churn",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "knn_model_loaded": "knn_churn" in models}


def as_customer_frame(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data[CUSTOMER_FEATURES].copy()

    return pd.DataFrame(data, columns=CUSTOMER_FEATURES)


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data = as_customer_frame(data)
    data["high_spender"] = (data["MonthlyCharges"] > 80).astype(int)
    data["is_new_customer"] = (data["tenure"] < 12).astype(int)
    return data


def predict_probabilities(data: pd.DataFrame) -> pd.Series:
    knn = models["knn_churn"]
    return pd.Series(knn.predict_proba(add_engineered_features(data))[:, 1])


def calculate_shap_values(customer: Customer) -> dict[str, float]:
    explainer = models["shap_explainer"]
    customer_data = pd.DataFrame([customer.model_dump()])[CUSTOMER_FEATURES]
    shap_values = explainer.shap_values(customer_data, nsamples="auto")[0]

    return {
        feature: round(float(value), 4)
        for feature, value in zip(CUSTOMER_FEATURES, shap_values, strict=True)
    }


@app.post("/predict")
def predict(customer: Customer) -> Prediction:
    knn = models.get("knn_churn")
    if knn is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    X = pd.DataFrame([customer.model_dump()])
    proba = float(knn.predict_proba(X)[0, 1])

    return Prediction(
        churn=proba >= settings.decision_threshold,
        churn_probability=round(proba, 4),
        shap_values=calculate_shap_values(customer),
    )


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]) -> list[Prediction]:
    if not customers:
        raise HTTPException(
            status_code=422, detail="batch must contain at least one customer"
        )

    knn = models.get("knn_churn")
    if knn is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    X = pd.DataFrame([customer.model_dump() for customer in customers])
    probabilities = knn.predict_proba(X)[:, 1]

    return [
        Prediction(
            churn=float(proba) >= settings.decision_threshold,
            churn_probability=round(float(proba), 4),
        )
        for proba in probabilities
    ]
