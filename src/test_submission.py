import pandas as pd
import numpy as np

def generateSubmission(
            results_df,
            metric_col,
            models_pipeline,
            X_train,
            y_train,
            test_dataset,
            path: str,
            row_id_col: str = 'row_id',
      ):
      """ Trains the best model and exports predicted probabilities to a CSV file.

      Args:
            results_df (pd.DataFrame): Results DataFrame for model selection.
            metric_col (str): Column name of the metric used to choose the best model.
            models_pipeline (dict): Dictionary of candidate pipelines, with keys = model name.
            X_train (pd.DataFrame): training DataFrame.
            y_train(pd.Series): training target Series.
            test_dataset (pd.DataFrame): Dataframe of unseen observations.
            path (str): export path for the submission file.
            row_id_col (str): Column name containing unique row identifiers.

      Returns:
            None: Exports a CSV file with columns: [row_id, predicted_probabilities].
      """
      best_model = models_pipeline[results_df.loc[results_df[metric_col].idxmax(), 'Model']]
      best_model.fit(X_train, y_train)

      row_ids = test_dataset['row_id']
      test_dataset = test_dataset.drop(columns=[row_id_col], errors='ignore')

      test_probs = best_model.predict_proba(test_dataset)[:, 1]

      submission_df = pd.DataFrame({
            'row_id': row_ids,
            'predicted_probability': np.round(test_probs, 4)
      })

      submission_df.to_csv(path, index=False)