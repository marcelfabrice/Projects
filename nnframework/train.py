import numpy as np
import inference as inf
from evaluate import test_samples
from sklearn.datasets import load_digits
from data_loader import load_data, load_data_scaled

X, y = load_digits(return_X_y=True)

X_train, y_train, X_test, y_test = load_data(X, y)

net = inf.Sequential(
    inf.Linear(64, 16),
    inf.ReLU(),
    inf.Linear(16, 10),
    inf.Softmax()
)
mse = inf.CrossEntropy()
opt = inf.Optim.Adam(net.parameters(), 0.001)

net.fit(
    X_train=X_train,
    y_train=y_train, 
    X_test=X_test,
    y_test=y_test,
    criteron=mse, 
    optimizer=opt, 
    epochs=25, 
    verbose=True, 
    classification=True, 
    testing=True
)

net.evaluate(X_test, y_test, mse, True)