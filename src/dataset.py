import pandas as pd
from sklearn.model_selection import train_test_split

def returnDataset(path, leakage_columns=None):
      df = pd.read_csv(path)
      df = df.drop(leakage_columns, axis=1)
      return df