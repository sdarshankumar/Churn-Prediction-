# Customer Churn Prediction Platform

End-to-end churn prediction pipeline: EDA → feature engineering → SMOTE →
XGBoost (tuned) → SHAP interpretability → simulated retention-strategy
business impact → Streamlit demo.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV at:
```
data/raw/telco.csv
```

## Run order

```bash
cd src
python train.py             # trains, tunes, saves model + SHAP chart to ../models and ../reports
python business_impact.py   # runs retention simulation, exports scored customer CSV
```

Then, from the project root (not from inside `app/`):

```bash
streamlit run app/streamlit_app.py
```

## Folder structure

```
churn-prediction/
├── data/raw/telco.csv          <- add this yourself
├── src/
│   ├── preprocess.py           <- cleaning + feature engineering
│   ├── train.py                <- split, SMOTE, train, tune, SHAP, save model
│   └── business_impact.py      <- retention simulation, scored CSV export
├── models/                     <- populated by train.py
├── reports/                    <- populated by train.py / business_impact.py
├── app/streamlit_app.py        <- interactive demo
├── notebooks/01_eda.ipynb      <- exploratory analysis
└── requirements.txt
```

## Outputs

- `models/churn_model.pkl` — trained XGBoost model
- `models/feature_columns.pkl` — exact feature order used at training time
- `reports/model_comparison.csv` — Logistic Regression / Random Forest / XGBoost comparison
- `reports/shap_summary.png` — global feature importance chart
- `reports/scored_customers.csv` — every test customer with churn probability + risk tier, ready to import into Power BI / Tableau
