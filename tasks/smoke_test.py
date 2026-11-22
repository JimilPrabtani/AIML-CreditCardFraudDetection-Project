"""Smoke test (inference only, never trains). Run: python tasks/smoke_test.py"""
from pathlib import Path
import json
import joblib
import pandas as pd

ART = Path("artifacts")
EXPECTED = [f"V{i}" for i in range(1, 29)] + ["Amount"]

pipe = joblib.load(ART / "fraud_pipeline.pkl")
thr = float(json.loads((ART / "threshold.json").read_text())["threshold"])
sample = pd.read_csv(ART / "sample_test_rows.csv")
assert all(c in sample.columns for c in EXPECTED), "sample rows missing columns"

proba = pipe.predict_proba(sample[EXPECTED])[:, 1]
assert ((proba >= 0) & (proba <= 1)).all(), "probabilities out of range"
n_flag = int((proba >= thr).sum())
print(f"OK: {len(sample)} rows scored, threshold={thr}, flagged={n_flag}")

# batch path: mimic app validation (drop id/Class, coerce numeric)
df = sample.copy()
for c in ["id", "Class", "pred", "proba"]:
    if c in df.columns:
        df = df.drop(columns=[c])
assert not df[EXPECTED].isna().any().any(), "non-numeric values in batch"
print("OK: batch validation path")
