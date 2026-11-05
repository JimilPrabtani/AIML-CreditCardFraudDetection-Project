# Model Card: Credit Card Fraud Triage (2023 dataset)

**Data:** `creditcard_2023.csv` (568,630 rows, 2023 snapshot, Kaggle
`nelgiriyewithana/credit-card-fraud-detection-dataset-2023`). Columns `id`, `V1–V28`
(PCA-anonymized, no business meaning), `Amount`, `Class`. ~50/50 fraud/legit
(unlike production ~0.1%). Stratified 80/20 split, `random_state=42`, `id` dropped.
Local file, git-ignored.
**Model:** sklearn `Pipeline(StandardScaler(Amount) + RandomForest n_estimators=200,
max_depth=25, n_jobs=-1, random_state=42)`. Remainder (`V1–V28`) passed through
unscaled (already ~standardized). Frozen: `artifacts/fraud_pipeline.pkl` (37 MB).
App loads it with `joblib` and never retrains. Isolation Forest was evaluated as an
unsupervised contrast only and is not shipped.
**Threshold:** 0.85, cost-based (FP=$10 review, FN=$500 missed fraud) on a 50k-row
0.1% prod simulation (5 repeats, `tasks/fix_honesty.py`). The notebook's lab sweep
uses default 0.5; re-running the notebook overwrites `threshold.json`, so re-run
the fix script afterwards. See `artifacts/prod_metrics.json`,
`artifacts/threshold_cost_table.csv`, and the app's Performance page.
**Lab (113,726 test, 50/50):** F1 0.9998, ROC-AUC 1.0, PR-AUC 0.9999 @0.5; 20 FP, 0 FN.
**Prod-sim (50k, 0.1% fraud):** @0.5 prec 0.751/rec 1.0/cost 166; @0.85 prec 0.954/rec 1.0/cost 24.
**Top features:** V15 (0.217), V11 (0.129), V5 (0.112) by RF importance. Error slice:
FP Amount mean 9022 vs ok 12044.
**Intended use:** SOC triage demo only (batch review queue). Local Streamlit, in-memory scoring, no logging, no PII handling.
**Out of scope:** Production bank scoring, real-time blocking, explainable case narratives (blocked by PCA anonymization).
**Limits:** Single 2023 snapshot (drift certain); balanced lab overstates precision — do not
quote lab F1 as production accuracy; threshold assumes the stated cost matrix; needs
calibration + drift monitoring before any real deployment.
