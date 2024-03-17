import pandas as pd

# Step 1: Load the CSV
df = pd.read_csv('Dataset/feature_output_10k_final.csv', header=0)

# Step 2: Remove useless columns (index, id, text, label, stems)
columns_to_drop = [0, 1, 2, 4, 5, 14]  # 0-indexed columns to drop
df = df.drop(columns=df.columns[columns_to_drop])