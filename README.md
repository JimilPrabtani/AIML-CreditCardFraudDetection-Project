# Credit Card Fraud Detection (2023 Dataset)

University group ML project. Python + Jupyter + Streamlit, all local on PC.
Adds a SOC-style triage demo: frozen model scores transactions as FRAUD/LEGIT with a cost-based threshold.

## Dataset

`creditcard_2023.csv` — 568,630 European card transactions (2023).
https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023.
Columns: `id`, `V1–V28`, `Amount`, `Class (1=fraud, 0=legit)`.
Source: Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023`.
Balance: ~50% fraud / 50% legit (balanced, unlike production ~0.1%).
`V1–V28` are PCA-anonymized: useful for modelling, no business meaning.
Limitations: one 2023 snapshot (drift certain), anonymized features, balanced lab overstates precision.
The CSV (310 MB) is git-ignored; download it from Kaggle and place it next to `analysis_model.ipynb`.

## Setup (VS Code, Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

1. Open `analysis_model.ipynb`, select `.venv` kernel, Run All top to bottom (trains, compares 5 models, exports to `artifacts/`).
2. Re-apply the honesty fix (production simulation + cost-based threshold):
   ```powershell
   .venv\Scripts\python.exe tasks/fix_honesty.py
   ```
   This reuses the frozen pipeline (no retrain) and writes `artifacts/prod_metrics.json`,
   `artifacts/threshold_cost_table.csv`, and sets `artifacts/threshold.json` to 0.85.
   Note: re-running the notebook overwrites `threshold.json` with the lab default 0.5 —
   re-run step 2 afterwards.
3. Launch the triage app (loads the frozen pipeline, never trains):
   ```powershell
   .venv\Scripts\python.exe -m streamlit run app.py
   ```

## What the notebook does

1. Load + describe (dataframes for shape, missing, duplicates, balance).
2. EDA (balance bar, Amount by class, correlation-with-fraud tables, top-V plots).
3. Stratified 80/20 split, drop `id`, scale `Amount` only inside Pipeline (no leakage).
4. Logistic Regression baseline (confusion matrix + accuracy/precision/recall/F1/ROC-AUC/PR-AUC).
5. Bagging vs Random Forest comparison dataframe + ROC/PR overlay.
6. XGBoost + Isolation Forest (unsupervised anomaly contrast) comparison.
7. 5-fold CV + small RandomizedSearchCV on F1 (best: RF, 200 trees, max_depth 25).
8. Threshold sweep 0.20–0.80 as a dataframe + plot (lab default 0.5; shipped 0.85 comes from step 2 above).
9. Top-10 feature importance dataframe + FP/FN Amount analysis.
10. Export to `artifacts/` for the app (pipeline, threshold, sample rows).

## Honesty fix (why the threshold is 0.85, not 0.5)

Lab test is 50/50, so every threshold looks perfect (F1 ~0.999). `tasks/fix_honesty.py`
resamples the frozen model's test scores to a 50k-row, 0.1% fraud simulation (5 repeats)
and prices errors at FP=$10 review / FN=$500 missed fraud:

| threshold | prod precision | prod recall | cost / 50k txns |
| --- | --- | --- | --- |
| 0.50 | 0.751 | 1.0 | 166 |
| 0.85 (shipped) | 0.954 | 1.0 | 24 |

Full sweep: `artifacts/threshold_cost_table.csv`. The model is separable on this
same-distribution data, so real-world drift would hurt more — the app's Performance
page shows both views side by side instead of claiming production accuracy.

## The app (`app.py`, local only, no logging)

- Predict: pick a held-out sample row + Amount what-if slider (V1–V28 stay fixed,
  PCA values can't be typed by hand), or upload a batch CSV (`V1–V28`, `Amount`;
  `id`/`Class` columns are dropped with a warning). Output: `proba`, `decision`,
  `risk` (LOW / REVIEW / BLOCK) + scored-CSV download. In-memory only, nothing stored.
- Performance: lab metrics vs 0.1% prod-simulation tables from `artifacts/prod_metrics.json`.
- About: model card, intended SOC-triage use, limits.

## Models in one line

- Logistic Regression: one straight boundary, interpretable baseline.
- Bagging: many trees on random rows, majority vote, cuts variance.
- Random Forest: bagging + random columns per split, handles V interactions (shipped).
- XGBoost: trees fix previous errors sequentially, usually most accurate.
- Isolation Forest: no labels, flags rare outliers, expected weaker (noted as contrast only).

## Project structure

```
analysis_model.ipynb
app.py
creditcard_2023.csv            # local only, git-ignored (310 MB Kaggle download)
requirements.txt
MODEL_CARD.md
artifacts/fraud_pipeline.pkl   # frozen RF pipeline (37 MB)
artifacts/threshold.json       # {"threshold": 0.85, ...}
artifacts/prod_metrics.json
artifacts/threshold_cost_table.csv
artifacts/sample_test_rows.csv
tasks/plan.md
tasks/todo.md
tasks/fix_honesty.py
```

## Results

Best model: RandomForest (200 trees, max_depth 25). Lab (113k test, 50/50): F1 0.9998, ROC-AUC 1.0, PR-AUC 0.9999 @0.5 (20 FP, 0 FN).
Honesty fix: 50k-row 0.1% prod simulation — @0.5 prec 0.751/cost 166, @0.85 prec 0.954/rec 1.0/cost 24. Threshold shipped: 0.85 (see `artifacts/prod_metrics.json`).
Top features: V15, V11, V5. FP Amount mean 9022 vs ok 12044.
Limits: PCA V1–V28 (no semantics), 2023 snapshot (drift), balanced lab vs 0.1% prod. See `MODEL_CARD.md`.
