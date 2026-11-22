# Research paper (IEEE conference format)

Source: `root.tex` — IEEEtran, US Letter, two-column, matches
`IEEE-research-paper-referance.pdf` in the repo root.

## Compile (needs LaTeX once; no Python needed)

Overleaf: upload `root.tex` + `figs/` + `../images/` (or set `\graphicspath`)
and compile with pdfLaTeX (`IEEEtran.cls` is preinstalled on
Overleaf and in TeX Live / MiKTeX — no need to upload it).

```powershell
cd paper
pdflatex root.tex
bibtex root   # not needed: references are inline (thebibliography)
pdflatex root.tex
pdflatex root.tex
```

Output: `paper/root.pdf` — generated, committed, ready to submit
alongside the `root.tex` source.

Readable PDF without LaTeX: `tasks/make_paper_pdf.py` regenerates
`paper/root.pdf` (IEEE-styled two-column, same verified numbers and
`paper/figs/` figures) — already built and committed.

## What the paper reports (all numbers verified)

- Baselines @0.5, lab 113,726 rows 50/50: Table II in the tex
  (logreg / bagging / RF-200-d25 / XGBoost / IsolationForest —
  values from `analysis_model.ipynb` stored outputs).
- Prod simulation @0.1% fraud, 5x50k rows, FP=$10/FN=$500:
  `artifacts/prod_metrics.json` (threshold 0.85, cost $24 vs $166).
- Lab threshold sensitivity: same JSON (`lab_threshold_table`).
- Retrain ablations + latency: `tasks/audit_eval.py` →
  `artifacts/audit_metrics.json` (re-run after any retrain).

## If you re-run the notebook

The export cell resets `threshold.json` to 0.5 — always re-run
`tasks/fix_honesty.py` afterwards, then re-check the paper tables
against the regenerated JSON before submitting.
