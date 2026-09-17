import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def remove_outliers(df, columns, z_thresh=3):
    mask = pd.Series([True] * len(df))
    for col in columns:
        if df[col].std() == 0: continue
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        mask &= (z_scores < z_thresh)
    return df[mask]

def main():
    df = pd.read_csv("data/raw/wine.csv")
    
    # Rename column to avoid slash issues in JSON/API later
    df = df.rename(columns={'od280/od315_of_diluted_wines': 'od280_od315_of_diluted_wines'})
    
    # Clean: Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # Clean: Remove outliers
    feature_cols = [c for c in df.columns if c != 'target']
    df_clean = remove_outliers(df, feature_cols)
    
    # Split
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Processed data saved to data/processed/")

if __name__ == "__main__":
    main()