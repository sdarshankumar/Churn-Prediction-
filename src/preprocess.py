"""
preprocess.py
Loads the raw Telco churn CSV, cleans it, and engineers features.
Run standalone to sanity-check the output, or import load_and_clean()
and engineer_features() into train.py.
"""

import pandas as pd
import numpy as np


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw CSV and fix known data issues."""
    df = pd.read_csv(path)

    # TotalCharges is stored as text and has blank strings for
    # brand-new customers (tenure = 0) — coerce to numeric and fill
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # customerID is just an identifier, not a predictive feature
    df = df.drop('customerID', axis=1)

    # Target column: Yes/No -> 1/0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioral/engagement features, then one-hot encode categoricals."""

    # Average spend per month of tenure — a proxy for "value density"
    # +1 in denominator avoids divide-by-zero for tenure == 0
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Bucket tenure into human-readable ranges (useful for reporting too)
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
    )

    # Count how many "Yes" services each customer has subscribed to —
    # this is the "engagement" signal referenced in the project writeup
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = (df[service_cols] == 'Yes').sum(axis=1)

    # One-hot encode all remaining categorical columns (including
    # the tenure_bucket we just created)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.append('tenure_bucket')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


if __name__ == "__main__":
    # Quick standalone check: python src/preprocess.py
    RAW_PATH = "../data/raw/telco.csv"
    df = load_and_clean(RAW_PATH)
    df = engineer_features(df)
    print(f"Shape after cleaning + feature engineering: {df.shape}")
    print(df.head())
    print("\nChurn rate:", df['Churn'].mean().round(3))
