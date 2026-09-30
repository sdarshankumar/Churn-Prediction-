"""
business_impact.py
Loads the already-trained model, scores the test set, segments
customers into risk tiers, and simulates the effect of a targeted
retention campaign on the highest-risk segment.

Run AFTER train.py has produced ../models/churn_model.pkl
Run with: python src/business_impact.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from preprocess import load_and_clean, engineer_features

# ---------------------------------------------------------------------
# 1. Rebuild the exact same test set used during training
# ---------------------------------------------------------------------
# We must recreate the split with the same random_state so this test
# set matches the one train.py evaluated on.
RAW_PATH = "../data/raw/telco.csv"

df = load_and_clean(RAW_PATH)
df = engineer_features(df)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------
# 2. Load the trained model and score the test set
# ---------------------------------------------------------------------
model = joblib.load("../models/churn_model.pkl")
feature_columns = joblib.load("../models/feature_columns.pkl")

# Make sure column order matches what the model was trained on
X_test = X_test[feature_columns]

X_test_scored = X_test.copy()
X_test_scored['churn_probability'] = model.predict_proba(X_test)[:, 1]
X_test_scored['actual_churn'] = y_test.values

# ---------------------------------------------------------------------
# 3. Segment customers into risk tiers
# ---------------------------------------------------------------------
X_test_scored['risk_tier'] = pd.cut(
    X_test_scored['churn_probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

tier_counts = X_test_scored['risk_tier'].value_counts()
print("=== Customers per risk tier ===")
print(tier_counts)

# ---------------------------------------------------------------------
# 4. Simulate a targeted retention campaign
# ---------------------------------------------------------------------
# Assumption: a targeted retention offer (discount, proactive outreach,
# loyalty perk) reduces churn probability among High-risk customers by
# this relative amount. 30% is a reasonable, commonly-cited industry
# figure for well-targeted retention campaigns — state this assumption
# explicitly in your report, don't present it as an observed result.
INTERVENTION_EFFECT = 0.30

high_risk = X_test_scored[X_test_scored['risk_tier'] == 'High']
baseline_churners = high_risk['actual_churn'].sum()
simulated_prevented = baseline_churners * INTERVENTION_EFFECT

overall_churners = y_test.sum()
attrition_reduction_pct = (simulated_prevented / overall_churners) * 100

print("\n=== Retention campaign simulation ===")
print(f"High-risk customers in test set: {len(high_risk)}")
print(f"Actual churners within that group: {baseline_churners}")
print(f"Estimated churners prevented by targeted retention: {simulated_prevented:.1f}")
print(f"Estimated overall attrition reduction: {attrition_reduction_pct:.1f}%")

# ---------------------------------------------------------------------
# 5. Export scored customer list for the BI dashboard (Power BI / Tableau)
# ---------------------------------------------------------------------
X_test_scored.to_csv("../reports/scored_customers.csv", index=False)
print("\nSaved scored customer list to ../reports/scored_customers.csv")
print("Import this CSV into Power BI / Tableau to build the reporting dashboard.")
