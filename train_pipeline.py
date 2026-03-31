"""
ChurnIQ ML Pipeline
====================
1. Load & clean IBM Telco Customer Churn dataset
2. Engineer features (tenure_bucket, charge_ratio, service_count)
3. Add synthetic churn-correlated columns
4. Train XGBoost + LightGBM ensemble
5. Evaluate (AUC-ROC, AUC-PR, classification report)
6. Save models & enriched dataset
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_CSV = os.path.join(DATA_DIR, "telco.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "telco_clean.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  LOAD & CLEAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 60)
print("STEP 1: Loading & Cleaning Data")
print("=" * 60)

df = pd.read_csv(RAW_CSV)
print(f"  Raw shape: {df.shape}")

# TotalCharges has whitespace strings for new customers → coerce to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges with 0
missing_tc = df["TotalCharges"].isna().sum()
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print(f"  Filled {missing_tc} missing TotalCharges with 0")

# Drop customerID (not a feature)
df = df.drop(columns=["customerID"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  LABEL ENCODE ALL CATEGORICALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nSTEP 2: Encoding Categorical Columns")
print("-" * 60)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded '{col}' → {len(le.classes_)} classes")

print(f"\n  Encoded {len(cat_cols)} categorical columns")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nSTEP 3: Engineering Features")
print("-" * 60)

# 3a. tenure_bucket  – cut into 5 equal-width bins
df["tenure_bucket"] = pd.cut(
    df["tenure"], bins=5, labels=[0, 1, 2, 3, 4]
).astype(int)
print("  ✓ tenure_bucket (5 bins)")

# 3b. charge_ratio  – MonthlyCharges / TotalCharges
#     Guard against divide-by-zero (TotalCharges == 0 for new customers)
df["charge_ratio"] = np.where(
    df["TotalCharges"] > 0,
    df["MonthlyCharges"] / df["TotalCharges"],
    0.0,
)
print("  ✓ charge_ratio (MonthlyCharges / TotalCharges)")

# 3c. service_count  – sum of all Yes/No service columns
#     After label encoding, Yes maps to the highest integer for binary cols.
#     The original Yes/No service columns are:
service_cols = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# For each service column, we count it as "subscribed" when the encoded value
# corresponds to "Yes" in the original data.
service_count = pd.Series(0, index=df.index)
for col in service_cols:
    le = label_encoders[col]
    yes_code = list(le.classes_).index("Yes") if "Yes" in le.classes_ else None
    if yes_code is not None:
        service_count += (df[col] == yes_code).astype(int)

df["service_count"] = service_count
print(f"  ✓ service_count (sum of {len(service_cols)} service columns)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  SYNTHETIC CHURN-CORRELATED COLUMNS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\nSTEP 4: Adding Synthetic Churn-Correlated Features")
print("-" * 60)

churn_col = "Churn"
churn_mask = df[churn_col] == 1  # 1 = "Yes" after label encoding

n = len(df)

# 4a. support_ticket_sentiment: float in [-1, 1]
#     Churners skew negative, non-churners skew positive
df["support_ticket_sentiment"] = np.where(
    churn_mask,
    np.clip(np.random.normal(-0.4, 0.35, n), -1, 1),
    np.clip(np.random.normal(0.3, 0.35, n), -1, 1),
)
print("  ✓ support_ticket_sentiment (churners skew negative)")

# 4b. nps_score: int 0-10
#     Churners skew low (mean ~3), non-churners skew high (mean ~7)
df["nps_score"] = np.where(
    churn_mask,
    np.clip(np.random.normal(3, 1.5, n), 0, 10).astype(int),
    np.clip(np.random.normal(7, 1.5, n), 0, 10).astype(int),
)
print("  ✓ nps_score (churners skew low)")

# 4c. days_since_login: int 1-90
#     Churners skew high (mean ~55), non-churners skew low (mean ~10)
df["days_since_login"] = np.where(
    churn_mask,
    np.clip(np.random.normal(55, 18, n), 1, 90).astype(int),
    np.clip(np.random.normal(10, 8, n), 1, 90).astype(int),
)
print("  ✓ days_since_login (churners skew high)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  SAVE CLEANED ENRICHED DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df.to_csv(CLEAN_CSV, index=False)
print(f"\n  ✓ Saved enriched dataset → {CLEAN_CSV}")
print(f"  Final shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  TRAIN / TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("STEP 5: Train / Test Split")
print("=" * 60)

FEATURES = [c for c in df.columns if c != churn_col]
X = df[FEATURES]
y = df[churn_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
print(f"  Churn rate – Train: {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.  TRAIN XGBOOST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("STEP 6: Training XGBoost")
print("=" * 60)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
print(f"  XGBoost AUC-ROC: {roc_auc_score(y_test, xgb_proba):.4f}")
print(f"  XGBoost AUC-PR:  {average_precision_score(y_test, xgb_proba):.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8.  TRAIN LIGHTGBM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("STEP 7: Training LightGBM")
print("=" * 60)

lgbm_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    verbosity=-1,
)
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)
lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
print(f"  LightGBM AUC-ROC: {roc_auc_score(y_test, lgbm_proba):.4f}")
print(f"  LightGBM AUC-PR:  {average_precision_score(y_test, lgbm_proba):.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9.  ENSEMBLE (Average Probabilities)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print("STEP 8: Ensemble (XGBoost + LightGBM Average)")
print("=" * 60)

ensemble_proba = (xgb_proba + lgbm_proba) / 2
ensemble_preds = (ensemble_proba >= 0.5).astype(int)

auc_roc = roc_auc_score(y_test, ensemble_proba)
auc_pr = average_precision_score(y_test, ensemble_proba)

print(f"\n  ┌─────────────────────────────────────┐")
print(f"  │  ENSEMBLE AUC-ROC:  {auc_roc:.4f}          │")
print(f"  │  ENSEMBLE AUC-PR:   {auc_pr:.4f}          │")
print(f"  └─────────────────────────────────────┘")

print("\n  Classification Report (Ensemble):")
print("  " + "-" * 55)
report = classification_report(y_test, ensemble_preds, target_names=["No Churn", "Churn"])
for line in report.split("\n"):
    print(f"  {line}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. SAVE MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
xgb_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
lgbm_path = os.path.join(MODEL_DIR, "lgbm_model.pkl")

joblib.dump(xgb_model, xgb_path)
joblib.dump(lgbm_model, lgbm_path)

print("\n" + "=" * 60)
print("STEP 9: Models Saved")
print("=" * 60)
print(f"  ✓ XGBoost  → {xgb_path}")
print(f"  ✓ LightGBM → {lgbm_path}")
print(f"  ✓ Clean CSV → {CLEAN_CSV}")
print("\n✅ Pipeline complete!")
