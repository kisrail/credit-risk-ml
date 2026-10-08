from src.dataset import returnDataset
from src.train import get_model_pipelines, evaluate_model
from sklearn.model_selection import train_test_split
from src.test_submission import generateSubmission
import time

SEED = 42

LEAKAGE_COLS = [
      'principal_recv',
      'payments_total',
      'last_txn_amt',
      'late_fees',
      'residual_amt',
      'fee_adj',
      'review_gap_m',
      'account_flag',
      'score_recent'
]

METADATA_COLS = [
      'row_id',
      'app_kind'
]

start_time = time.perf_counter()

train_df = returnDataset('data/d2assignment_dataset.csv', LEAKAGE_COLS, METADATA_COLS)
test_df = returnDataset('data/d2assignment_test.csv', LEAKAGE_COLS)

X = train_df.drop('target', axis=1)
y = train_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)

PLACEHOLDER_COLS = [
      'm_since_delinq',
      'm_since_inquiry',
      'm_since_card_open'
]

OTHER_PLD_COLS = [
      'pct_cards_hi_util',
      'card_headroom'
]

MEDIAN_COLS = [
      'emp_years',
      'card_util',
      'revolving_util',
      'avg_balance'
]

SELECTED_FEATURES = [
      # Positive Correlated Featuers
      'pricing_rate',
      'risk_band',
      'new_accts_24m',
      'new_accts_12m',
      'debt_ratio',
      'inquiries_6m',
      'pct_cards_hi_util',
      'card_util',

      # Negative Correlated Features
      'newest_account_m',
      'mortgage_ct',
      'housing_status',
      'total_balance',
      'avg_balance',
      'card_headroom',
      'card_limit_total',
      'total_credit_limit',
      'bureau_score',

      # Engineered Features
      'avg_balance_per_tradeline',
      'monthly_payment_on_income',
      'free_cash_flow',
      'loan_income_ratio',

      'util_gap',
      'headroom_to_income',
      'limit_saturation',
      'non_mortgage_loans',

      'inquiry_intensity',
      'inquiry_acceleration',
      'acct_velocity_ratio',

      'delinq_per_tradeline',
      'total_derogatory_events',
      'dirty_tradelines'
]

DROP_COLS = [
      'region_code',
      'area_code'
]

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

pipelines = get_model_pipelines(
      PLACEHOLDER_COLS,
      OTHER_PLD_COLS,
      MEDIAN_COLS,
      SELECTED_FEATURES,
      DROP_COLS,
      scale_pos_weight=scale_pos_weight,
      seed=SEED
)
results = evaluate_model(pipelines, X_train, y_train, cv_splits=5, seed=SEED)

print(results)

generateSubmission(
      results_df=results,
      metric_col = 'Mean ROC-AUC',
      models_pipeline=pipelines,
      X_train=X,
      y_train=y,
      test_dataset=test_df,
      path='./submission.csv',
      row_id_col='row_id'
)

elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")