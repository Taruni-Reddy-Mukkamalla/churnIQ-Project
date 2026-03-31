"""
Cox Proportional Hazards survival analysis for customer churn.

Fits a CoxPH model using lifelines, then for each customer outputs:
  - median_days_to_churn (int)
  - survival probabilities at months 1, 2, 3, 6, 9, 12
Saves results to data/survival_predictions.csv.
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/telco_clean.csv")

# ── Define model variables ───────────────────────────────────────────────────
TIME_COL = "tenure"
EVENT_COL = "Churn"
COVARIATES = [
    "MonthlyCharges",
    "Contract",
    "support_ticket_sentiment",
    "nps_score",
    "days_since_login",
]

# Month → day conversion (approx 30 days per month)
MONTH_TIMEPOINTS = {1: 30, 2: 60, 3: 90, 6: 180, 9: 270, 12: 360}

# ── Prepare modelling dataframe ─────────────────────────────────────────────
# tenure is in months; convert to days for more granular survival curves
model_df = df[COVARIATES + [TIME_COL, EVENT_COL]].copy()
model_df[TIME_COL] = model_df[TIME_COL].clip(lower=1)  # avoid zero-tenure

# ── Fit Cox PH model ────────────────────────────────────────────────────────
cph = CoxPHFitter()
cph.fit(model_df, duration_col=TIME_COL, event_col=EVENT_COL)

print("=" * 60)
print("COX PROPORTIONAL HAZARDS  —  MODEL SUMMARY")
print("=" * 60)
cph.print_summary()

# ── Predict per-customer survival ────────────────────────────────────────────
# Survival function returns a DataFrame: rows = timepoints, cols = customers
surv_funcs = cph.predict_survival_function(model_df[COVARIATES])

# Median survival time per customer (in tenure units = months → convert to days)
median_months = cph.predict_median(model_df[COVARIATES])
# Replace inf (customers who never churn in the model) with a cap of 999 days
median_days = (median_months * 30).replace([np.inf, -np.inf], 999).astype(int)

# ── Extract survival probabilities at months 1,2,3,6,9,12 ───────────────────
# surv_funcs index is in tenure units (months). Interpolate at desired months.
month_points = [1, 2, 3, 6, 9, 12]

# Reindex to include the desired time points, interpolate, then pick them
all_times = sorted(set(surv_funcs.index.tolist() + month_points))
surv_reindexed = surv_funcs.reindex(all_times).interpolate(method="index")
surv_at_months = surv_reindexed.loc[month_points]  # shape: (6, n_customers)

# ── Assemble output DataFrame ────────────────────────────────────────────────
out = df.copy()
out["median_days_to_churn"] = median_days.values

for m in month_points:
    col_name = f"surv_prob_month_{m}"
    out[col_name] = surv_at_months.loc[m].values.round(4)

# Keep only the required new columns plus an identifier
output_cols = (
    list(df.columns)
    + ["median_days_to_churn"]
    + [f"surv_prob_month_{m}" for m in month_points]
)
out = out[output_cols]

# ── Save ─────────────────────────────────────────────────────────────────────
out.to_csv("data/survival_predictions.csv", index=False)
print(f"\n✅  Saved survival_predictions.csv  ({out.shape[0]} rows × {out.shape[1]} cols)")
print(out[["tenure", "Churn", "median_days_to_churn"] +
          [f"surv_prob_month_{m}" for m in month_points]].head(10))
