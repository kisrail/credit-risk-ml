from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def pipeline_imputers(placeholder_cols, median_cols, other_placeholder_cols):
      base_imputer = ColumnTransformer([
            ('negative_placeholder', SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True), placeholder_cols),
            ('zero_placeholder', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True), other_placeholder_cols),
            ('median_placeholder', SimpleImputer(strategy='median'), median_cols)
      ], remainder='passthrough', verbose_feature_names_out=False)

      return base_imputer

def feature_selector(features):
      selected_features = ColumnTransformer([
            ('feature_selection', 'passthrough', features)
      ], remainder='drop')

      return selected_features

def feature_engineering(df):
      df_out = df.copy()

      df_out['avg_balance_per_tradeline'] = df_out['avg_balance'] / (df_out['open_tradelines'] + 1e-5)
      df_out['inquiry_acceleration'] = df_out['inquiries_6m'] / (df_out['m_since_inquiry'] + 1e-5)
      df_out['delinq_per_tradeline'] = df_out['delinq_24m'] / (df_out['open_tradelines'] + 1e-5)
      df_out['monthly_payment_on_income'] = (df_out['monthly_payment'] * 12) / (df_out['income_annual'] + 1e-5)
      df_out['total_derogatory_events'] = df_out['public_records'] + df_out['bankruptcies'] + df_out['collections_12m']

      return df_out