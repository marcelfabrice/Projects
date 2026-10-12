import mlflow.data.dataset
from mlt.data.preprocessing import preprocessing_pipeline
from sklearn.neighbors import KNeighborsClassifier
from mlt.data.preprocessing import get_data
from mlt.models.metrics import evaluate, log_metrics
from mlt.config import INTERIM_DATA_DIR
from sklearn.pipeline import Pipeline
from mlt.config import get_logger
import mlflow.sklearn
import logging
import mlflow
import yaml


def fit_model(hyperparams: dict):
    logger = get_logger(__name__)

    preprocessor = preprocessing_pipeline(logger=logger)

    with mlflow.start_run(run_name="knn-classifier-run", nested=True):
        # get hyperparameters from config

        mlflow.log_params(hyperparams)
        logger.log(logging.INFO, "params loaded")

        # get cleaned data
        X, y = get_data(logger=logger, INTERIM_DATA_DIR=INTERIM_DATA_DIR)

        # K-Nearest-Neighbor Classifier
        k_nearest_neighbour = KNeighborsClassifier(**hyperparams)

        # simple pipe
        pipe = Pipeline([("preprocessor", preprocessor), ("knn", k_nearest_neighbour)])

        # 5 fold on unsplitted data -> crossvalidate
        scores = evaluate(estimator=pipe, X=X, y=y)

        log_metrics(scores)

        logger.info(
            f"roc auc cross validation score: {scores['test_roc_auc'].mean():.3f}"
        )
        logger.info(f"best score: {scores['test_roc_auc'].max():.3f}")
        logger.info(f"worst score: {scores['test_roc_auc'].min():.3f}")

        pipe.fit(X, y)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            registered_model_name="customer-churn-classifier",
            artifact_path="customer-churn-classifier",
        )

    return scores[
        "test_roc_auc"
    ].mean()  # return metric to be maximized in hyperparameter optimization


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    hyperparams = {
        "n_neighbors": params["model"]["k"],
        "weights": params["model"]["weights"],
        "metric": params["model"]["metric"],
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("customer-churn-classifier")

    fit_model(hyperparams=hyperparams)
