# TODO: Fraud Triage App — DONE (all tasks built 2026-11-05)

## Task 1: Prod-simulation + cost threshold — done
- [x] `tasks/fix_honesty.py` resamples frozen test scores to 50k-row 0.1% sim (5 repeats), prices FP=$10/FN=$500
- [x] `artifacts/threshold.json` = 0.85, `artifacts/prod_metrics.json` + `artifacts/threshold_cost_table.csv` written
- [x] No retrain of `best_rf`

## Task 2: Conclusion + model card — done
- [x] Notebook conclusion cell filled (no `[INSERT]` remains); `MODEL_CARD.md` written

## Task 3: App loader + validation — done (`app.py`)
- [x] Cached load with clear error + `st.stop()` on missing artifact
- [x] Schema validator: drops `id`/`Class` with warning, coerces numeric, rejects missing/non-numeric
- [x] `streamlit run app.py` boots offline (verified :8501)

## Task 4: Single-sample predict — done
- [x] Library picker from `sample_test_rows.csv` + Amount what-if slider, prob/decision/risk + distance-to-threshold
- [x] No hand-typed V1-V28 form

## Task 5: Batch CSV predict + download — done
- [x] Upload → validate → `predict_proba` → table with prob/decision/risk + flagged counts + scored-CSV download
- [x] In-memory only note

## Task 6: Performance + About pages — done
- [x] Numbers load from `prod_metrics.json` (match `fix_honesty.py` exactly); lab-vs-prod warning banner present
- [x] About: data dict, limits, SOC use, no-PII note

## Task 7: Resume polish — done
- [x] `requirements.txt` pinned to `.venv` versions; README rewritten (run steps, honesty table, structure, results, limits)
- [x] Fresh-install path documented (`pip install -r requirements.txt`)

## Checkpoint: Complete
- [x] Notebook → `tasks/fix_honesty.py` → app runs; README has real numbers + limits. Regenerate: re-run notebook, then `fix_honesty.py` (notebook export resets threshold to 0.5).
