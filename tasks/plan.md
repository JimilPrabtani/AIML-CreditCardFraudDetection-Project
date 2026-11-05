# Implementation Plan: Honest Fraud Triage App (2023 dataset, frozen model)

## Overview
Keep `analysis_model.ipynb` training frozen. Fix the `F1 0.9999 / ROC-AUC 1.0` red flag by adding an honest production-simulation layer (balanced lab vs 0.1% prod), picking a cost-based threshold, and shipping a local-only Streamlit SOC-triage app that loads `artifacts/fraud_pipeline.pkl + threshold.json` and never retrains. Inputs work around PCA `V1-V28` (batch CSV + sampled library rows, no hand-typed V's). Output is `P(fraud) + decision + risk band + explanation`.

## Scenario
SOC analyst pastes/uploads recent transactions. App scores them, flags REVIEW/BLOCK, shows why (prob distance, Amount context, global top features V15/V11/V5), and shows lab-vs-prod performance so demo is not misleading. Local PC only, no data leaves machine, no logging.

## Architecture Decisions
- `best_rf` (RandomForest from `RandomizedSearchCV`, currently in `artifacts/fraud_pipeline.pkl`) stays the AI model. No new dataset, no new features. Rationale: university constraint + keeps notebook valid.
- Threshold moves from hardcoded `0.5` (`analysis_model.ipynb:2441`, `artifacts/threshold.json`) to cost-based (`FN=$500, FP=$10` default, adjustable in app display only, persisted value updated once). Rationale: cyber relevance > F1 on balanced data.
- App never trains: `joblib.load` pipeline (includes Amount scaler). Validates 29 cols `V1-V28 + Amount`, rejects `id/Class` leakage cols with clear error.
- Two input paths only (no 28-box manual form): (1) batch CSV upload, (2) pick from `sample_test_rows.csv` + Amount what-if slider (perturbs Amount, keeps V's). Rationale: V's are PCA, untypable.
- Honesty panel mandatory: lab (50/50) vs simulated prod (0.1% fraud: precision collapse, alerts/day). Plus model card limits (anonymized, 2023 snapshot, drift).
- Optional second opinion: existing IsolationForest noted as weak; show only as anomaly flag, not primary decision, to avoid scope creep.

## Task List

### Phase 1: Honesty fix (notebook appendix only, no rewrite)
- [ ] Task 1: Prod-simulation + cost threshold (append 1-2 cells, update `threshold.json` + metrics JSON)
- [ ] Task 2: Conclusion + model card fill (replace `[INSERT]`s, add limits paragraph)

### Checkpoint: Honesty fix
- [ ] Balanced vs 0.1%-sim tables render, cost curve picks threshold, artifacts updated

### Phase 2: App foundation
- [ ] Task 3: `app.py` loader + validation (cached load, schema check, threshold display)
- [ ] Task 4: Single-sample predict (library picker + Amount slider + prob/decision/risk output)

### Checkpoint: Foundation
- [ ] `streamlit run app.py` scores 1 row end-to-end offline

### Phase 3: Batch + performance
- [ ] Task 5: Batch CSV predict + download (table, counts, FP/FN estimate at prod rate)
- [ ] Task 6: Performance + About pages (load saved metrics, lab-vs-prod, limits, no-logging note)

### Checkpoint: Core app
- [ ] Batch + single + performance pages work from frozen artifacts

### Phase 4: Resume polish
- [ ] Task 7: `requirements.txt`, README results, demo GIF, `tasks/` evidence

### Checkpoint: Complete
- [ ] Fresh `.venv` install -> notebook appendix -> app runs; README has real numbers + limits

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Trivial separability can't be "fixed" by tuning | High | Frame as lab-vs-prod; don't claim prod accuracy; use cost/recall@FP-rate |
| Manual V1-V28 input meaningless | High | No manual form; batch + sampled rows + Amount what-if only |
| Threshold 0.5 misleading | Med | Cost-based threshold, persisted once, slider is display-only in app |
| 568k-row rerun slow | Med | Appendix cells reuse `best_rf`/`X_test`, no full retrain |
| Overclaim in resume | Med | Model card + README limits section required |

## Open Questions (for you before build)
- Cost defaults OK? `FN=$500, FP=$10` or your bank assumption?
- Prod rate to simulate? Default `0.1%` fraud.
- Keep IsolationForest second opinion or hide it? Default: hide, mention in About.
- App name for resume? Default: `Fraud Triage Console (local)`.
