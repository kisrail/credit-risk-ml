import pandas as pd

def returnDataset(path: str, leakage_cols=None, metadata_cols=None):
      """ Returns a dataset.

      Args:
            path (str): import path of the dataset.
            leakage_cols (arr): array with leakage features from the dataset.
            metadata_cols (arr): array with unnecessary columns from the dataset.

      Returns:
            Pandas DataFrame.
      
      """
      df = pd.read_csv(path)
      leakage_cols = leakage_cols or []
      metadata_cols = metadata_cols or []
      df = df.drop([*leakage_cols, *metadata_cols], axis=1)
      return df