# data_sampling.py
def sample_data(df, n=None, random_state=None):
    if n is not None:
        df_sample = df.sample(n=n, random_state=random_state)
    else:
        df_sample = df
    return df_sample
