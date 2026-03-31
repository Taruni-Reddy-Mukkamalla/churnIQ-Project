"""
SHAP Explanations Generator
============================
Loads the XGBoost model and telco_clean.csv, computes SHAP values
for every customer, extracts top-6 features, and saves a JSON file
with customer_id, churn_prob, plain_english_summary, and top_features.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLEAN_CSV = os.path.join(DATA_DIR, "telco_clean.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
OUTPUT_PATH = os.path.join(DATA_DIR, "shap_explanations.json")

# ── Feature name → plain-English label mapping ──────────────────────────────
FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen status",
    "Partner": "Having a partner",
    "Dependents": "Having dependents",
    "tenure": "Tenure (months with provider)",
    "PhoneService": "Phone service subscription",
    "MultipleLines": "Multiple phone lines",
    "InternetService": "Internet service type",
    "OnlineSecurity": "Online security add-on",
    "OnlineBackup": "Online backup add-on",
    "DeviceProtection": "Device protection plan",
    "TechSupport": "Tech support plan",
    "StreamingTV": "Streaming TV service",
    "StreamingMovies": "Streaming movies service",
    "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges to date",
    "tenure_bucket": "Tenure bracket",
    "charge_ratio": "Monthly-to-total charge ratio",
    "service_count": "Number of subscribed services",
    "support_ticket_sentiment": "Support ticket sentiment",
    "nps_score": "NPS (Net Promoter) score",
    "days_since_login": "Days since last login",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  LOAD DATA & MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 65)
print("STEP 1: Loading model & data")
print("=" * 65)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(CLEAN_CSV)

TARGET = "Churn"
FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES]

print(f"  Model loaded from:  {MODEL_PATH}")
print(f"  Data loaded from:   {CLEAN_CSV}")
print(f"  Customers:          {len(df)}")
print(f"  Features:           {len(FEATURES)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  COMPUTE SHAP VALUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("STEP 2: Computing SHAP values (TreeExplainer)")
print("=" * 65)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print(f"  SHAP matrix shape:  {shap_values.shape}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  PREDICTED CHURN PROBABILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
churn_probs = model.predict_proba(X)[:, 1]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  BUILD JSON OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("STEP 3: Building per-customer SHAP explanations")
print("=" * 65)

TOP_K = 6
results = []

for idx in range(len(df)):
    customer_id = f"CUST-{idx + 1:04d}"
    prob = float(round(churn_probs[idx], 4))
    sv = shap_values[idx]

    # Top-6 features by absolute SHAP value
    top_indices = np.argsort(np.abs(sv))[::-1][:TOP_K]

    top_features = []
    for fi in top_indices:
        feat_name = FEATURES[fi]
        shap_val = float(round(sv[fi], 4))
        direction = "increases churn risk" if shap_val > 0 else "decreases churn risk"
        top_features.append({
            "feature": feat_name,
            "shap_value": shap_val,
            "direction": direction,
        })

    # ── Plain-English summary ────────────────────────────────────────────
    risk_level = (
        "very high" if prob >= 0.8 else
        "high" if prob >= 0.6 else
        "moderate" if prob >= 0.4 else
        "low" if prob >= 0.2 else
        "very low"
    )

    # Build driver phrases from top-3 features
    driver_phrases = []
    for tf in top_features[:3]:
        label = FEATURE_LABELS.get(tf["feature"], tf["feature"])
        verb = "raising" if tf["direction"] == "increases churn risk" else "lowering"
        driver_phrases.append(f"{label} ({verb} risk)")

    summary = (
        f"This customer has a {risk_level} churn probability of {prob:.0%}. "
        f"The key drivers are: {', '.join(driver_phrases)}."
    )

    results.append({
        "customer_id": customer_id,
        "churn_prob": prob,
        "plain_english_summary": summary,
        "top_features": top_features,
    })

    # Progress indicator
    if (idx + 1) % 1000 == 0 or idx == len(df) - 1:
        print(f"  Processed {idx + 1:,} / {len(df):,} customers")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  SAVE JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 65}")
print(f"✅  Saved {len(results):,} SHAP explanations → {OUTPUT_PATH}")
print(f"{'=' * 65}")

# ── Preview first 2 entries ──────────────────────────────────────────────────
print("\n📋 Preview (first 2 customers):\n")
for entry in results[:2]:
    print(json.dumps(entry, indent=2))
    print()
