from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def pipeline_imputers(placeholder_cols, other_placeholder_cols, median_cols):
      base_imputer = ColumnTransformer([
            ('negative_placeholder', SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True), placeholder_cols),
            ('zero_placeholder', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True), other_placeholder_cols),
            ('median_placeholder', SimpleImputer(strategy='median'), median_cols)
      ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')

      return base_imputer

def feature_selector(features):
      selected_features = ColumnTransformer([
            ('feature_selection', 'passthrough', features)
      ], remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')

      return selected_features

def drop_features(drop_cols):
      features_to_drop = ColumnTransformer([
            ('drop_features', 'drop', drop_cols)
      ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')

      return features_to_drop