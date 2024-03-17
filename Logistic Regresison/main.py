import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


########################### DATA CLEANING ###########################
# Load the CSV
df = pd.read_csv('Dataset/feature_output_10k_final.csv', header=0)

# Remove useless columns
columns_to_drop = [0, 1, 2, 4, 5, 14]  # 0-indexed columns to drop
df = df.drop(columns=df.columns[columns_to_drop])


########################### DATA SAMPLING ###########################
# Take a sample of n rows
# df_sample = df.sample(n=1000, random_state=28)
df_sample = df


########################### SELECTING FEATURES/LABELS ###########################
y = df_sample.iloc[:, 0]  # First column is y
X = df_sample.iloc[:, 1:]  # All columns except the first one are X

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########################### MODEL TRAINING ###########################
# Build and train a logistic regression model
log_reg_model = log_reg_model = LogisticRegression( C=3, max_iter=10000)
log_reg_model.fit(X_train, y_train)


########################### CALCULATE METRICS ###########################
y_pred = log_reg_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)



########################### Visualisation ###########################

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Plot bar chart for precision, recall, and F1 score
metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)
plt.show()