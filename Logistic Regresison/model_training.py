# model_training.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# def train_model(X, y, test_size=0.2, random_state=28):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
#     log_reg_model = LogisticRegression(C=1, max_iter=10000)
#     log_reg_model.fit(X_train, y_train)
#     return log_reg_model, X_test, y_test

def train_model(X, y, test_size=0.3, random_state=28):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define a grid of hyperparameters to search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'max_iter': [1000, 5000, 10000]  # Maximum number of iterations
    }

    # Create a logistic regression classifier
    log_reg = LogisticRegression()

    # Instantiate GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train a logistic regression model with the best hyperparameters
    best_log_reg_model = LogisticRegression(**best_params)
    best_log_reg_model.fit(X_train, y_train)

    return best_log_reg_model, X_test, y_test