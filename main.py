from src.dataset import returnDataset
from src.train import get_model_pipelines, evaluate_model
from sklearn.model_selection import train_test_split
from src.test_submission import generateSubmission

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
      'row_id'
]

train_df = returnDataset('data/raw/d2assignment_dataset.csv', LEAKAGE_COLS, METADATA_COLS)
test_df = returnDataset('data/raw/d2assignment_test.csv', LEAKAGE_COLS)

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
      'risk_band',
      'bureau_score',
      'card_limit_total',
      'avg_balance_per_tradeline',
      'inquiry_acceleration',
      'delinq_per_tradeline',
      'monthly_payment_on_income',
      'total_derogatory_events',
      'emp_years',
      'income_annual',
      'debt_ratio'
]

scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

pipelines = get_model_pipelines(
      PLACEHOLDER_COLS,
      OTHER_PLD_COLS,
      MEDIAN_COLS,
      SELECTED_FEATURES,
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