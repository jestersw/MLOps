import pandas as pd
from sklearn.datasets import load_wine
import os
import numpy as np

def main():
    os.makedirs("data/raw", exist_ok=True)
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Inject dummy missing values and outliers to demonstrate cleaning
    df.loc[0, 'alcohol'] = np.nan
    df.loc[1, 'malic_acid'] = 1000.0 
    
    df.to_csv("data/raw/wine.csv", index=False)
    print("Raw data saved to data/raw/wine.csv")

if __name__ == "__main__":
    main()