# main.py
import pandas as pd
from data_cleaning import load_dataset
from data_sampling import sample_data
from model_training import train_model
from metrics_evaluation import evaluate_metrics
from visualisation import plot_confusion_matrix, plot_metrics_bar_chart
import matplotlib.pyplot as plt

# Load the dataset
df = load_dataset('Dataset/feature_output_10k_final.csv')

test_df = load_dataset('Paraphrase/ai_human_with_features.csv')

# Sample the data
# df_sample = sample_data(df,n=1000,random_state=28)
df_sample = df

# optionally perform RFE


# Select features and labels
y = df.generated  # First column is y
X = df.drop(columns = ['generated'])  # All columns except the first one are X

test_y = test_df.generated  # First column is y
test_X = test_df.drop(columns = ['generated'])  # All columns except the first one are X

# Train the model
log_reg_model, X_test, y_test = train_model(X, y)

# Evaluate metrics
y_pred_stand = log_reg_model.predict(X_test)
accuracy_s, precision_s, recall_s, f1_s, conf_matrix_s = evaluate_metrics(y_test, y_pred_stand)
print("Accuracy:", accuracy_s)
print("Precision:", precision_s)
print("Recall:", recall_s)
print("F1 Score:", f1_s)
print("Confusion Matrix:")
print(conf_matrix_s)

# Visualize the results
plot_confusion_matrix(conf_matrix_s)
plot_metrics_bar_chart(['Precision', 'Recall', 'F1 Score'], [precision_s, recall_s, f1_s])

print("=====================================================================")

#evaluate against paraphrased
y_pred = log_reg_model.predict(test_X)
accuracy, precision, recall, f1, conf_matrix = evaluate_metrics(test_y, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the results
plot_confusion_matrix(conf_matrix)
plot_metrics_bar_chart(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1])
