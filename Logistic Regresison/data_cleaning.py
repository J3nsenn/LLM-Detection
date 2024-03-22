# data_cleaning.py
import pandas as pd



def load_dataset(file_path):
    df = pd.read_csv(file_path, header=0)
    
    features_to_divide = ['no_discourse_markers','no_pronouns','grammatical_errors','named_entity_counts']
    df[features_to_divide] = df[features_to_divide].div(df['length'],axis=0)
    columns_to_drop = [0, 1, 2, 4, 5, 14] 
    df = df.drop(columns=df.columns[columns_to_drop])
    return df




