def feature_engineering(df):
      """ Returns the dataset with additional engineered features.

      Args:
            df (pd.DataFrame)

      Returns:
            Pandas DataFrame.

      """
      df_out = df.copy()

      # (1) Payment Affordability
      df_out['avg_balance_per_tradeline'] = df_out['avg_balance'] / (df_out['open_tradelines'] + 1e-5)
      df_out['monthly_payment_on_income'] = (df_out['monthly_payment'] * 12) / (df_out['income_annual'] + 1e-5) # Monthly installment burden relative to annual income.
      df_out['free_cash_flow'] = (df_out['income_annual'] / 12) * (1 - df_out['debt_ratio']) - df_out['monthly_payment'] # Monthly surplus remaining after satisfying debt obligations and loan installments.
      df_out['loan_income_ratio'] = df_out['principal_req'] / df_out['income_annual']

      # (2) Utilization
      df_out['util_gap'] = df_out['card_util'] - df_out['revolving_util']
      df_out['headroom_to_income'] = df_out['card_headroom'] / df_out['income_annual'] # Available liquidity against annual income.
      df_out['limit_saturation'] = df_out['total_balance'] / df_out['total_credit_limit'] # Exposure ratio across all active accounts.
      df_out['non_mortgage_loans'] = df_out['balance_ex_mortgage'] / (df_out['total_balance'] + 1) # Share of unsecured loans on total balance.

      # (3) Credit Card Seeking Behavior
      df_out['inquiry_intensity'] = df_out['inquiries_6m'] / (df_out['new_accts_12m'] + 1)
      df_out['inquiry_acceleration'] = df_out['inquiries_6m'] / (df_out['m_since_inquiry'] + 1e-5)
      df_out['acct_velocity_ratio'] = df_out['new_accts_12m'] / (df_out['total_tradelines'] + 1)

      # (4) Delinquency History
      df_out['delinq_per_tradeline'] = df_out['delinq_24m'] / (df_out['open_tradelines'] + 1e-5)
      df_out['total_derogatory_events'] = df_out['public_records'] + df_out['bankruptcies'] + df_out['collections_12m']
      df_out['dirty_tradelines'] = df_out['total_tradelines'] * (1 - df_out['pct_clean_history']) # Count of historically troubled credit lines.

      return df_out