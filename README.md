# Credit Card Fraud Detection (2023 Dataset)

University group ML project. Python + Jupyter + Streamlit, all local on PC.

## Dataset

`creditcard_2023.csv` — 568,630 European card transactions (2023).
https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023.
Columns: `id`, `V1–V28`, `Amount`, `Class (1=fraud, 0=legit)`.
Source: Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023`.
Balance: ~50% fraud / 50% legit (balanced, unlike production ~0.1%).
`V1–V28` are PCA-anonymized: useful for modelling, no business meaning.
Limitations: one 2023 snapshot (drift certain), no cost matrix, anonymized features.

## Setup (VS Code, Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib streamlit jupyter
```

Place `creditcard_2023.csv` next to `analysis_model.ipynb`.

## Run

1. Open `analysis_model.ipynb`, select `.venv` kernel, Run All top to bottom.
2. Each step: read markdown → run code → read dataframe output.
3. Fill `[INSERT]` numbers in the conclusion markdown for the report.
4. Exported files appear in `artifacts/` (pipeline, threshold, sample rows).

## What the notebook does

1. Load + describe (dataframes for shape, missing, duplicates, balance).
2. EDA (balance bar, Amount by class, correlation-with-fraud tables, top-V plots).
3. Stratified 80/20 split, drop `id`, scale `Amount` only inside Pipeline (no leakage).
4. Logistic Regression baseline (confusion matrix + accuracy/precision/recall/F1/ROC-AUC/PR-AUC).
5. Bagging vs Random Forest comparison dataframe + ROC/PR overlay.
6. XGBoost + Isolation Forest (unsupervised anomaly contrast) comparison.
7. 5-fold CV + small RandomizedSearchCV on F1.
8. Threshold sweep 0.20–0.80 as a dataframe + plot (default 0.5).
9. Top-10 feature importance dataframe + FP/FN Amount analysis.
10. Export to `artifacts/` for the app.

## Models in one line

- Logistic Regression: one straight boundary, interpretable baseline.
- Bagging: many trees on random rows, majority vote, cuts variance.
- Random Forest: bagging + random columns per split, handles V interactions.
- XGBoost: trees fix previous errors sequentially, usually most accurate.
- Isolation Forest: no labels, flags rare outliers, expected weaker.

## Project structure

```
analysis_model.ipynb
creditcard_2023.csv
artifacts/fraud_pipeline.pkl
artifacts/threshold.json
artifacts/sample_test_rows.csv
README.md
```

## Results (fill after run)

Best model: [INSERT], F1: [INSERT], ROC-AUC: [INSERT], PR-AUC: [INSERT], threshold: [INSERT].
Top features: [INSERT]. Main errors: [INSERT].

## Streamlit (next step)

Load `artifacts/fraud_pipeline.pkl` + `threshold.json`, never retrain.
Pages: Predict (CSV batch + random held-out row), Performance (saved tables/plots), About.
V1–V28 cannot be typed by hand, demo only, no data logging.

# Credit Card Fraud Detection (2023 Dataset)

University group ML project. Python + Jupyter + Streamlit, all local on PC.

## Dataset

`creditcard_2023.csv` — 568,630 European card transactions (2023).
Columns: `id`, `V1–V28`, `Amount`, `Class (1=fraud, 0=legit)`.
Source: Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023`.
Balance: ~50% fraud / 50% legit (balanced, unlike production ~0.1%).
`V1–V28` are PCA-anonymized: useful for modelling, no business meaning.
Limitations: one 2023 snapshot (drift certain), no cost matrix, anonymized features.

## Setup (VS Code, Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib streamlit jupyter
```

Place `creditcard_2023.csv` next to `analysis_model.ipynb`.

## Run

1. Open `analysis_model.ipynb`, select `.venv` kernel, Run All top to bottom.
2. Each step: read markdown → run code → read dataframe output.
3. Fill `[INSERT]` numbers in the conclusion markdown for the report.
4. Exported files appear in `artifacts/` (pipeline, threshold, sample rows).

## What the notebook does

1. Load + describe (dataframes for shape, missing, duplicates, balance).
2. EDA (balance bar, Amount by class, correlation-with-fraud tables, top-V plots).
3. Stratified 80/20 split, drop `id`, scale `Amount` only inside Pipeline (no leakage).
4. Logistic Regression baseline (confusion matrix + accuracy/precision/recall/F1/ROC-AUC/PR-AUC).
5. Bagging vs Random Forest comparison dataframe + ROC/PR overlay.
6. XGBoost + Isolation Forest (unsupervised anomaly contrast) comparison.
7. 5-fold CV + small RandomizedSearchCV on F1.
8. Threshold sweep 0.20–0.80 as a dataframe + plot (default 0.5).
9. Top-10 feature importance dataframe + FP/FN Amount analysis.
10. Export to `artifacts/` for the app.

## Models in one line

- Logistic Regression: one straight boundary, interpretable baseline.
- Bagging: many trees on random rows, majority vote, cuts variance.
- Random Forest: bagging + random columns per split, handles V interactions.
- XGBoost: trees fix previous errors sequentially, usually most accurate.
- Isolation Forest: no labels, flags rare outliers, expected weaker.

## Project structure

```
analysis_model.ipynb
creditcard_2023.csv
artifacts/fraud_pipeline.pkl
artifacts/threshold.json
artifacts/sample_test_rows.csv
README.md
```

## Results (fill after run)

Best model: [INSERT], F1: [INSERT], ROC-AUC: [INSERT], PR-AUC: [INSERT], threshold: [INSERT].
Top features: [INSERT]. Main errors: [INSERT].

## Streamlit (next step)

Load `artifacts/fraud_pipeline.pkl` + `threshold.json`, never retrain.
Pages: Predict (CSV batch + random held-out row), Performance (saved tables/plots), About.
V1–V28 cannot be typed by hand, demo only, no data logging.
