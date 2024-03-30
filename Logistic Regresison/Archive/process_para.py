import pandas as pd
from data_cleaning import load_dataset
from data_sampling import sample_data
from model_training import train_model
from metrics_evaluation import evaluate_metrics
from visualisation import plot_confusion_matrix, plot_metrics_bar_chart



test_df = load_dataset('Paraphrase/ai_human_with_features.csv')

test_y = test_df.generated  # First column is y
test_X = test_df.drop(columns = ['generated'])  # All columns except the first one are X



print(test_df.columns)