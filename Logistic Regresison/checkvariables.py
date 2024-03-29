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

feature_names = test_X.columns

importance = log_reg_model.coef_[0]

# summarize feature importance
for i, (feature_name, score) in enumerate(zip(feature_names, importance)):
    print('Feature: %s, Score: %.5f' % (feature_name, score))

# plot feature importance
plt.bar(feature_names, importance)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

for i, v in enumerate(importance):
    plt.text(i, v, "%.5f" % v, ha='center', va='bottom')

plt.show()

# Select the column corresponding to the 'perplexity' feature
perplexity_values = test_X['Perplexity Score']

# Plot histogram
plt.hist(perplexity_values, bins=20, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Perplexity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Perplexity Score')
plt.grid(True)
plt.show()

perplexity_values2 = X_test['Perplexity Score']

# Plot histogram
plt.hist(perplexity_values2, bins=20, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Perplexity Score')
plt.ylabel('Frequency')
plt.title('Distribution of Perplexity Score')
plt.grid(True)
plt.show()