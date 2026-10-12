# Models

This folder contains the model training, evaluation, and hyperparameter optimization code.

## Files

- `baseline_fit.py`: trains and logs the logistic regression baseline
- `model_fit.py`: trains and logs the KNN classifier
- `hpo.py`: runs Optuna hyperparameter optimization for KNN
- `metrics.py`: helper functions for saving model metric plots

## K-Nearest Neighbor

The main model is a k-nearest-neighbor classifier. It is evaluated with stratified cross validation and optimized with Optuna.

## KNN Weaknesses

- Slow on large datasets, because every test data point is compared to every training data point
- Affected by the curse of dimensionality, because distances lose impact when the feature space gets too large
- Sensitive to feature scales, so scaling is important to prevent large numeric values from dominating the distance calculation
- Sensitive to outliers, especially with a small `k`
- High memory usage, because the full training data has to be kept available
- Imbalanced data can be problematic, because the overrepresented class can dominate neighbor votes

## KNN Strengths

- Simple algorithm that is easy to understand
- Lazy learner, so there is no expensive training phase
- Can be used for classification and regression
- Supports non-linear decision boundaries
- Can perform well on small datasets
