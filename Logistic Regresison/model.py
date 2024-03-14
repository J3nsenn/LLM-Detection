from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def select_features(X_train, y_train, n_features=5):
    log_reg_model = LogisticRegression()
    rfe = RFE(estimator=log_reg_model, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    X_train_selected = rfe.transform(X_train)
    return X_train_selected

def tune_hyperparameters(X_train, y_train):
    log_reg_model = LogisticRegression()
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    grid_search = GridSearchCV(estimator=log_reg_model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_model
