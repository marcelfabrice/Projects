from mlt.data.preprocessing import preprocessing_pipeline
from mlt.data.preprocessing import get_data
from sklearn.neighbors import KNeighborsClassifier
from mlt.models.model_fit import fit_model
from mlt.models.metrics import evaluate, log_metrics
from mlt.config import INTERIM_DATA_DIR, MODELS_DIR
from sklearn.pipeline import Pipeline
from mlt.config import get_logger
import joblib
import optuna
import mlflow
import yaml

logger = get_logger(__name__)
with open("params.yaml") as f:
    params = yaml.safe_load(f)


def objective(trial: optuna.Trial):
    hyperparams = {
        "n_neighbors": trial.suggest_int(
            name="n_neighbors",
            low=params["hpo"]["k-lower"],
            high=params["hpo"]["k-upper"],
        ),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        # "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]), manhatten distance takes very long to compute
    }

    roc_auc_cv = fit_model(hyperparams=hyperparams)
    return roc_auc_cv


def hyperparameter_optimization():
    # ä TODO outer cv nested cv and treshhold opt
    with mlflow.start_run(run_name="knn-optuna-run"):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=params["hpo"]["trials"])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_auc", study.best_value)

        # get best model
        best_model = KNeighborsClassifier(**study.best_params)
        preprocessing = preprocessing_pipeline(logger=logger)
        best_pipe = Pipeline(
            [("preprocessing", preprocessing), ("best_knn", best_model)]
        )

        # track metrices

        X, y = get_data(logger=logger, INTERIM_DATA_DIR=INTERIM_DATA_DIR)

        scores = evaluate(estimator=best_pipe, X=X, y=y)

        # save model to mlflow
        best_pipe.fit(X, y)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "best_knn_model.joblib"
        joblib.dump(best_pipe, model_path)

        logger.info(f"saved best model to {model_path}")

        log_metrics(scores)
        mlflow.log_artifact(str(model_path), artifact_path="joblib_model")
        mlflow.sklearn.log_model(
            best_pipe,
            registered_model_name="best_knn_model",
            artifact_path="best_knn_model",
        )


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("k-nearest-neighbor-hbo")

    hyperparameter_optimization()
