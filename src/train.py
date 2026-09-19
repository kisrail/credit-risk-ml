import numpy as np
from src.feature_engineering import feature_engineering, pipeline_imputers
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model_pipelines(
      placeholder_cols: list,
      other_placeholder_cols: list,
      median_cols: list,
      #feature_cols: list,
      scale_pos_weight: float = 1.0,
      seed: int = 42            
) -> dict:
      model_pipelines = {
            'Logistic Regression (Baseline)': Pipeline([
                  ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),
                  ('base_imputer', pipeline_imputers(placeholder_cols, other_placeholder_cols, median_cols)),
                  ('remaining_imputers', SimpleImputer(strategy='median')),
                  ('scaler', StandardScaler()),
                  ('model', LogisticRegression(
                        max_iter=200,
                        class_weight='balanced',
                        random_state=seed
                  ))
            ]),
            """
            'Logistic Regression (Selected Features)': Pipeline([
                  ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),
                  ('base_imputer', pipeline_imputers(placeholder_cols, median_cols, other_placeholder_cols)),
                  ('remaining_imputers', SimpleImputer(strategy='median')),
                  ('scaler', StandardScaler()),
                  ('feature_selection', feature_selector(feature_cols)),
                  ('model', LogisticRegression(
                        max_iter=200,
                        class_weight='balanced',
                        random_state=SEED
                  ))
            ]),
            """
            'Random Forest': Pipeline([
                  ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),
                  ('base_imputer', pipeline_imputers(placeholder_cols, other_placeholder_cols, median_cols)),
                  ('remaining_imputers', SimpleImputer(strategy='median')),
                  ('model', RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        max_features='sqrt',
                        class_weight='balanced',
                        random_state=seed
                  ))
            ]),
            'XGBoost': Pipeline([
                  ('feature_engineering', FunctionTransformer(feature_engineering, validate=False)),
                  ('base_imputer', pipeline_imputers(placeholder_cols, other_placeholder_cols, median_cols)),
                  ('remaining_imputers', SimpleImputer(strategy='median')),
                  ('model', XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.03,
                        eval_metric='auc',
                        tree_method='hist',
                        scale_pos_weight=scale_pos_weight,
                        random_state=seed
                  ))
            ])
      }

      return model_pipelines

def evaluate_model(pipelines: dict, X_train, y_train, cv_splits: int = 5, seed: int = 42) -> dict:
      skf = StratifiedKFold(n_splits=cv_splits, random_state=seed, shuffle=True)
      results = {}

      for name, model in pipelines.items():
            cv_scores = cross_val_score(
                  estimator=model,
                  X=X_train,
                  y=y_train,
                  cv=skf,
                  scoring='roc_auc'
            )

            results[name] = {
                  'fold_scores': cv_scores,
                  'mean_auc': np.round(cv_scores.mean(), 4),
                  'std_auc': np.round(cv_scores.std(), 4)
            }

      return results