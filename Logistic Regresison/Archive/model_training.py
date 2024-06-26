# model_training.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, make_scorer

def train_modelunscaled(X, y, test_size=0.2, random_state=28):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Define a grid of hyperparameters to search
    param_grid = {
        'C': [0.001,0.01,0.1,1,10,100,1000,10000],  # Regularization parameter
        'max_iter': [100,500,1000, 5000, 10000],  # Maximum number of iterations
        # 'solver': ['liblinear','sag','lbfgs', 'saga', 'newton-cg']  # Solver algorithm
    }

    # Create a logistic regression classifier
    log_reg = LogisticRegression(random_state=28)

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

def train_model(X, y, test_size=0.2, random_state=28):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define a grid of hyperparameters to search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Regularization parameter
        'max_iter': [100, 500, 1000, 5000, 10000],  # Maximum number of iterations
        # 'solver': ['liblinear', 'sag', 'lbfgs', 'saga', 'newton-cg']  # Solver algorithm
    }

    # Create a logistic regression classifier
    log_reg = LogisticRegression(random_state=28,max_iter=10000)

    # Make scorer
    scorer = make_scorer(f1_score, average='weighted')

    # Instantiate GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring=scorer)

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train a logistic regression model with the best hyperparameters
    best_log_reg_model = LogisticRegression(**best_params,random_state=28)
    best_log_reg_model.fit(X_train_scaled, y_train)

    return best_log_reg_model, X_test_scaled, y_test


def train_model_rob(X, y, test_size=0.2, random_state=28):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = RobustScaler()

    # Fit the scaler on the training data and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the testing data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Define a grid of hyperparameters to search
    param_grid = {
        'C': [7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30],  # Regularization parameter
        'max_iter': [1000, 5000, 10000],  # Maximum number of iterations
        # 'solver': ['liblinear','sag','lbfgs', 'saga', 'newton-cg']  # Solver algorithm
    }

    # Create a logistic regression classifier
    log_reg = LogisticRegression()

    # Instantiate GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train a logistic regression model with the best hyperparameters
    best_log_reg_model = LogisticRegression(**best_params,random_state=28)
    best_log_reg_model.fit(X_train_scaled, y_train)

    return best_log_reg_model, X_test_scaled, y_test

def train_modelRFE(X, y, test_size=0.2, random_state=28, n_features_to_select=3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the testing data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Perform RFE if the number of features to select is specified
    if n_features_to_select is not None:
            # Create a logistic regression estimator
            log_reg = LogisticRegression()

            # Perform RFE to select features
            rfe = RFE(log_reg, n_features_to_select=n_features_to_select)
            rfe.fit(X_train_scaled, y_train)

            # Select the features
            selected_features = rfe.support_
            selected_feature_names = X.columns[selected_features]
            print("Selected Features:", selected_feature_names)
            
            # Update the training and testing data with selected features
            X_train_scaled = X_train_scaled[:, selected_features]
            X_test_scaled = X_test_scaled[:, selected_features]
    # Define a grid of hyperparameters to search
    param_grid = {
        'C': [7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30],  # Regularization parameter
        'max_iter': [1000, 5000, 10000],  # Maximum number of iterations
        # 'solver': ['liblinear','sag','lbfgs', 'saga', 'newton-cg']  # Solver algorithm
    }

    # Create a logistic regression classifier
    log_reg = LogisticRegression()

    # Instantiate GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_scaled, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train a logistic regression model with the best hyperparameters
    best_log_reg_model = LogisticRegression(**best_params)
    best_log_reg_model.fit(X_train_scaled, y_train)

    return best_log_reg_model, X_test_scaled, y_test