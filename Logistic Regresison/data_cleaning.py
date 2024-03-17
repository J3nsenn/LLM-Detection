# data_cleaning.py
import pandas as pd



def load_dataset(file_path, drop_columns):
    df = pd.read_csv(file_path, header=0)
    df = df.drop(columns=df.columns[drop_columns])
    return df


