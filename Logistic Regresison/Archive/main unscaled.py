# main.py
import pandas as pd
from data_cleaning import load_dataset
from data_sampling import sample_data
from model_training import train_modelunscaled
from metrics_evaluation import evaluate_metrics
from visualisation import plot_confusion_matrix, plot_metrics_bar_chart

# Load the dataset
df = load_dataset('Dataset/feature_output_10k_final.csv')

# Sample the data
# df_sample = sample_data(df,n=1000,random_state=28)
df_sample = df

# optionally perform RFE


# Select features and labels
y = df.generated  # First column is y
X = df.drop(columns = ['generated'])  # All columns except the first one are X

# Train the model
log_reg_model, X_test, y_test = train_modelunscaled(X, y)

# Evaluate metrics
y_pred = log_reg_model.predict(X_test)
accuracy, precision, recall, f1, conf_matrix = evaluate_metrics(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the results
plot_confusion_matrix(conf_matrix)
plot_metrics_bar_chart(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1])