"""
train.py
Full training pipeline:
  1. Load + clean + feature-engineer the data
  2. Train/test split with SMOTE on the training set only
  3. Train 3 candidate models and compare them
  4. Hyperparameter-tune the best one (XGBoost)
  5. Explain it with SHAP
  6. Save the final model to disk

Run with: python src/train.py
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

from preprocess import load_and_clean, engineer_features

# ---------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------
RAW_PATH = "../data/raw/telco.csv"
MODEL_OUT_PATH = "../models/churn_model.pkl"

df = load_and_clean(RAW_PATH)
df = engineer_features(df)

X = df.drop('Churn', axis=1)
y = df['Churn']

# ---------------------------------------------------------------------
# 2. Train/test split + SMOTE
# ---------------------------------------------------------------------
# stratify=y keeps the churn ratio consistent between train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# IMPORTANT: fit SMOTE only on the training set. Never resample the
# test set — it must reflect the real-world class distribution.
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Original training set churn rate: {y_train.mean():.3f}")
print(f"Resampled training set churn rate: {y_train_res.mean():.3f}")

# ---------------------------------------------------------------------
# 3. Train and compare baseline models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        eval_metric='logloss', random_state=42
    )
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    fitted_models[name] = model

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs)
    })

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n=== Model comparison ===")
print(results_df.to_string(index=False))

# Save this table — you'll put it directly in your report
results_df.to_csv("../reports/model_comparison.csv", index=False)

# ---------------------------------------------------------------------
# 4. Hyperparameter tuning on the best model (XGBoost)
# ---------------------------------------------------------------------
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

search = RandomizedSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train_res, y_train_res)

print("\n=== Best hyperparameters ===")
print(search.best_params_)
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

best_model = search.best_estimator_

# Re-evaluate the tuned model on the held-out test set
tuned_preds = best_model.predict(X_test)
tuned_probs = best_model.predict_proba(X_test)[:, 1]

print("\n=== Tuned XGBoost — test set performance ===")
print(f"Accuracy:  {accuracy_score(y_test, tuned_preds):.4f}")
print(f"Precision: {precision_score(y_test, tuned_preds):.4f}")
print(f"Recall:    {recall_score(y_test, tuned_preds):.4f}")
print(f"F1:        {f1_score(y_test, tuned_preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, tuned_probs):.4f}")

# ---------------------------------------------------------------------
# 5. Model interpretability with SHAP
# ---------------------------------------------------------------------
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance — save as an image for your report
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../reports/shap_summary.png", dpi=150)
plt.close()
print("\nSaved SHAP summary plot to ../reports/shap_summary.png")

# ---------------------------------------------------------------------
# 6. Save the final model + the exact feature column order
# ---------------------------------------------------------------------
# Saving the column list is critical — predict.py / the Streamlit app
# must build feature vectors in this exact order and with these
# exact one-hot columns, or predictions will be silently wrong.
joblib.dump(best_model, MODEL_OUT_PATH)
joblib.dump(list(X.columns), "../models/feature_columns.pkl")

print(f"\nSaved trained model to {MODEL_OUT_PATH}")
