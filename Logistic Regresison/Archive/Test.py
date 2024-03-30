import pandas as pd

df = pd.read_csv('Dataset/feature_output_10k_final.csv', header=0)
df = df.drop(df.columns[:2], axis=1)

df_sample = df.sample(n=20, random_state=28)
print(df_sample.iloc)
print(df_sample.dtypes)


# Define X (features) and Y (labels)
X = df_sample.iloc[:, 2:].values  # Features start from the third column
y = df_sample.iloc[:, 1].values    # Labels are in the second column

