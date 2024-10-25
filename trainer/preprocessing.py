
import argparse
import os
import warnings

import pandas as pd
import numpy as np

def reduce_memory_use(df):
    for f in df.columns:
        if "float" in str(df.dtypes[f]):
            if (df[f].min()>np.finfo(np.float16).min) & (df[f].max()<np.finfo(np.float16).max):
                df[f] = df[f].astype(np.float16)
    return df

def impute_missing_features(df):
    print("Imputing missing value by a forward fill, followed by a backward fill.")
    all_features = [c for c in df.columns if "feature_" in c]
    df[all_features] = df.groupby("symbol_id")[all_features].ffill()
    df[all_features] = df.groupby("symbol_id")[all_features].bfill()
    
    return df

def exclude_missing_days(df):
    exclude_days = range(529)
    return df[~df["date_id"].isin(exclude_days)]

def preprocess_train(df):
    print(f"Memory (Before): {df.memory_usage().sum()/1024**2: .2f}")
    df = reduce_memory_use(df)
    df = exclude_missing_days(df)
    df = impute_missing_features(df)
    df = reduce_memory_use(df)
    print(f"Memory (After):  {df.memory_usage().sum()/1024**2: .2f}")
    
    return df

def save_as_npz(df, output_file_path):
    np.savez_compressed(output_file_path, **{col: df[col].values for col in df.columns})
    return

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument()
    return parser.parse_args()

if __name__=="__main__":
    # args = parse_args()
    
    # Load dataset
    input_dir = "/opt/ml/processing/input"
    filename = "train_530.parquet/"
    input_data_path = os.path.join(input_dir, filename)
    print(os.listdir(input_dir))
    df = pd.read_parquet(input_data_path)
    print(f"Input dataframe shape: {df.shape}")
    
    # Preprocess
    df = preprocess_train(df)
    
    # Save dataset
    output_dir = "/opt/ml/processing/output"
    filename = "df_530_lite.npz"
    output_data_path = os.path.join(output_dir, filename)
    save_as_npz(df, output_data_path)
    print(f"Saving compressed npz at: {output_data_path}")
    