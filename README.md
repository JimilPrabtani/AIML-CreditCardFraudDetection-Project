# Credit Card Fraud Detection — SOC Triage Demo

Local ML demo: a frozen RandomForest scores 2023 card transactions as **FRAUD / LEGIT** with a cost-based threshold. Built for a university project, styled as a SOC review queue.

> Lab accuracy looks perfect (F1 ~0.999) because the test set is 50/50 balanced. Real fraud is ~0.1% — the app shows both views side by side so the demo doesn't mislead.

![Stack](/stack.png)

## Dataset

- File: `creditcard_2023.csv` (568,630 rows, local only, git-ignored — download from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) and place next to the notebook)
- Columns: `id`, `V1–V28` (PCA-anonymized, no business meaning), `Amount`, `Class` (1 = fraud)
- Split: stratified 80/20, `random_state=42`, `id` dropped, `Amount` scaled inside the Pipeline (no leakage)

## Results

Best model: **RandomForest (200 trees, max_depth 25)**. Full details in [`MODEL_CARD.md`](MODEL_CARD.md).

| View | Metric |
|---|---|
| Lab test (113k rows, 50/50) @0.5 | F1 0.9998, ROC-AUC 1.0, PR-AUC 0.9999 (20 FP, 0 FN) |
| Prod simulation (50k rows, 0.1% fraud) @0.5 | precision 0.751, recall 1.0, cost $166 |
| Prod simulation (50k rows, 0.1% fraud) @**0.85 shipped** | precision **0.954**, recall 1.0, cost **$24** |

Threshold 0.85 comes from `tasks/fix_honesty.py`: resamples frozen test scores to 5× 50k-row 0.1% simulations, prices errors at FP = $10 review / FN = $500 missed fraud, picks minimum cost. Top features: V15, V11, V5.

## Quickstart (Windows, VS Code)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
# 1. Train + export (open in VS Code, select .venv kernel, Run All)
#    analysis_model.ipynb  ->  writes artifacts/fraud_pipeline.pkl + sample rows

# 2. Set the honest threshold (re-run whenever the notebook is re-run)
.venv\Scripts\python.exe tasks/fix_honesty.py

# 3. Launch the app (loads frozen model, never trains)
.venv\Scripts\python.exe -m streamlit run app.py
```

> The notebook export resets `threshold.json` to 0.5 — step 2 sets it back to 0.85.

## Using the app (local only, nothing stored)

- **Score** — pick a held-out sample row + an Amount what-if slider (V1–V28 stay fixed, PCA values can't be typed by hand), or upload a batch CSV with `V1–V28, Amount` (`id`/`Class` are dropped with a warning). Output: fraud probability, FRAUD/LEGIT decision, LOW / REVIEW / BLOCK risk, plus scored-CSV download.
- **Performance** — lab metrics vs 0.1% prod-simulation tables from `artifacts/prod_metrics.json`.
- **About** — model card summary, intended use, limits.

## Project structure

```
analysis_model.ipynb        # train + compare 5 models + export to artifacts/
app.py                      # Streamlit triage console (inference only)
tasks/fix_honesty.py         # cost-based threshold + prod simulation (no retrain)
MODEL_CARD.md               # model card, intended use, limits
artifacts/
  fraud_pipeline.pkl        # frozen RF pipeline (37 MB, git-ignored if large)
  threshold.json            # {"threshold": 0.85, ...}
  prod_metrics.json         # lab + prod-sim tables the app renders
  sample_test_rows.csv      # held-out sample rows for the demo picker
requirements.txt
```

## Limits

- Single 2023 snapshot — drift is certain, do not quote lab F1 as production accuracy.
- `V1–V28` are anonymized — no case-level explanation possible; `Amount` is the only interpretable knob.
- Threshold assumes FP = $10 / FN = $500 — change the costs, change the threshold.
- Demo only: not a bank production system, no real-time blocking, no PII handling beyond in-memory scoring.
