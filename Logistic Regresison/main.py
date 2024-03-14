import numpy as np
from sklearn.model_selection import train_test_split
from model import select_features, tune_hyperparameters
from visualization import plot_results
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load or define X (features) and y (labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

# Select features
X_train_selected = select_features(X_train, y_train)

# Tune hyperparameters
best_model = tune_hyperparameters(X_train_selected, y_train)

# Subset X_test based on selected features
X_test_selected = X_test[:, best_model.get_support()]

# Predict on the testing set using the best model
y_pred = best_model.predict(X_test_selected)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot ROC curve for the optimized model
y_scores_opt = best_model.decision_function(X_test_selected)
plot_results(y_test, y_pred, y_scores_opt, label='Optimized')

# Show the plot
plt.show()
