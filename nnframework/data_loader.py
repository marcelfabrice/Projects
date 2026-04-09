import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(X, y):
    _X, _y = X.copy(), y.copy()
    X_train, X_test, y_train, y_test = train_test_split(_X,_y,test_size=0.2,random_state=42)
    return X_train, y_train, X_test, y_test


def load_data_scaled(X, y, test_size=0.2, shuffle=True):
    # In NumPy konvertieren
    X = np.array(X)
    y = np.array(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle
    )

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
