"""
Causal Uplift Modelling – S-Learner approach
=============================================
Uses scikit-uplift + LightGBM to estimate the individual treatment effect
(ITE) for every customer.

Treatment is synthetically assigned: 30 % of customers are flagged as
treated (e.g. received a retention offer).

Output
------
- data/uplift_scores.csv   – customerID-level uplift scores
- Console printout of Top-10 persuadable customers
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklift.models import SoloModel          # S-Learner wrapper
from sklift.metrics import uplift_auc_score

# ── 1. Load data ─────────────────────────────────────────────────────────
df = pd.read_csv("data/telco_clean.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# ── 2. Synthetic treatment assignment (30 % treated) ────────────────────
np.random.seed(42)
df["treatment"] = np.random.binomial(1, 0.30, size=len(df))
print(f"Treatment split → treated: {df['treatment'].sum()}, "
      f"control: {(df['treatment'] == 0).sum()}")

# ── 3. Feature / target split ───────────────────────────────────────────
target_col   = "Churn"
treat_col    = "treatment"
drop_cols    = [target_col, treat_col]

X = df.drop(columns=drop_cols)
y = df[target_col]
treatment = df[treat_col]

# Train / test split (stratify on target × treatment)
strat_key = y.astype(str) + "_" + treatment.astype(str)
X_train, X_test, y_train, y_test, trt_train, trt_test = train_test_split(
    X, y, treatment,
    test_size=0.25,
    random_state=42,
    stratify=strat_key,
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 4. S-Learner (SoloModel in scikit-uplift) ───────────────────────────
base_estimator = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

s_learner = SoloModel(estimator=base_estimator)
s_learner.fit(X_train, y_train, trt_train)
print("S-Learner trained ✓")

# ── 5. Predict uplift on TEST set & evaluate ────────────────────────────
uplift_test = s_learner.predict(X_test)

try:
    auc = uplift_auc_score(y_test, uplift_test, trt_test)
    print(f"Uplift AUC (test): {auc:.4f}")
except Exception as e:
    print(f"Uplift AUC could not be computed: {e}")

# ── 6. Score ALL customers ──────────────────────────────────────────────
uplift_all = s_learner.predict(X)

results = df.copy()
results["uplift_score"] = uplift_all
results.to_csv("data/uplift_scores.csv", index=False)
print(f"\n✅  Uplift scores saved → data/uplift_scores.csv  ({len(results)} rows)")

# ── 7. Top-10 persuadable customers (highest positive uplift) ───────────
top10 = (
    results
    .nlargest(10, "uplift_score")
    [["tenure", "MonthlyCharges", "Contract", "Churn",
      "treatment", "uplift_score"]]
    .reset_index()
    .rename(columns={"index": "customer_index"})
)

print("\n🏆  Top-10 Persuadable Customers (highest positive uplift score):")
print(top10.to_string(index=False))
