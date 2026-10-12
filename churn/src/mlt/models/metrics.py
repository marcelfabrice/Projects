import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate


def save_metric_curves(fpr, tpr, auc, reports_dir) -> None:
    plt.plot(fpr, tpr, label=f"CV ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Kurve aus Cross Validation")
    plt.legend()
    plt.savefig(reports_dir + "auc_roc.png")


def evaluate(estimator, X, y) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }

    return cross_validate(estimator=estimator, X=X, y=y, cv=cv, scoring=scoring)


def log_metrics(scores: dict) -> dict:
    metrics = {
        "roc_auc_mean": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "f1_mean": scores["test_f1"].mean(),
        "f1_std": scores["test_f1"].std(),
        "balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
        "balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
    }

    mlflow.log_metrics(metrics)
    return metrics
