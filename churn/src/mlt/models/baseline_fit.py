from mlt.data.preprocessing import get_data, preprocessing_pipeline
from sklearn.linear_model import LogisticRegression
from mlt.models.metrics import evaluate, log_metrics
from mlt.config import INTERIM_DATA_DIR, get_logger
from sklearn.pipeline import Pipeline
import logging
import mlflow
import yaml

logger = get_logger(__name__)
with open("params.yaml") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer-churn-baseline")

preprocessor = preprocessing_pipeline(logger=logger)

with mlflow.start_run(run_name="baseline-run"):
    # parameters for data split
    split_params = {
        "test_split": params["split"]["test_size"],
        "random_state": params["split"]["random_state"],
    }

    # parameters for baseline model
    baseline_params = {
        "max_iter": params["baseline"]["max_iter"],
        "class_weight": "balanced",
    }

    mlflow.log_params(split_params)
    mlflow.log_params(baseline_params)

    X, y = get_data(logger=logger, INTERIM_DATA_DIR=INTERIM_DATA_DIR)

    logging.info("deploying pipeline")

    baseline = LogisticRegression(**baseline_params)

    # model pipeline
    pipe = Pipeline(
        [
            ("preprocessing", preprocessor),
            ("baseline", baseline),
        ]
    )

    # fit baseline
    pipe.fit(X, y)

    logging.info("baseline fit")
    logging.info("baseline evaluation...")

    # 5 fold on unsplitted data -> crossvalidate
    scores = evaluate(estimator=pipe, X=X, y=y)

    log_metrics(scores)

    pipe.fit(X, y)
    mlflow.sklearn.log_model(
        sk_model=pipe,
        registered_model_name="customer-churn-baseline",
        artifact_path="customer-churn-baseline",
    )
