"""Task 1: honesty fix - prod simulation + cost threshold. Appends nothing, just recomputes from frozen pipeline."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

RANDOM_STATE = 42
COST_FP = 10
COST_FN = 500
PROD_FRAUD_RATE = 0.001
N_PROD = 50000

df = pd.read_csv("creditcard_2023.csv")
X = df.drop(columns=["Class", "id"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pipe = joblib.load("artifacts/fraud_pipeline.pkl")
proba = pipe.predict_proba(X_test)[:, 1]

def metrics_at(y_true, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "thr": round(float(thr), 2),
        "prec": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "rec": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
    }

# Lab (balanced) sweep
ths = np.arange(0.1, 0.91, 0.05)
lab_rows = [metrics_at(y_test.values, proba, t) for t in ths]
lab_f1 = f1_score(y_test, proba >= 0.5)
lab_roc = roc_auc_score(y_test, proba)
lab_pr = average_precision_score(y_test, proba)

# Prod simulation: 5 repeats of N_PROD rows at 0.1% fraud
rng = np.random.RandomState(RANDOM_STATE)
X_test_v = X_test.reset_index(drop=True)
y_v = y_test.reset_index(drop=True)
fraud_idx = np.where(y_v.values == 1)[0]
legit_idx = np.where(y_v.values == 0)[0]
n_fraud = int(N_PROD * PROD_FRAUD_RATE)
n_legit = N_PROD - n_fraud
prod_thr_rows = []
for t in ths:
    precs, recs, f1s, costs = [], [], [], []
    for rep in range(5):
        fi = rng.choice(fraud_idx, size=n_fraud, replace=False)
        li = rng.choice(legit_idx, size=n_legit, replace=False)
        idx = np.concatenate([fi, li])
        yt = y_v.values[idx]
        yp = proba[idx]
        pred = (yp >= t).astype(int)
        precs.append(precision_score(yt, pred, zero_division=0))
        recs.append(recall_score(yt, pred, zero_division=0))
        f1s.append(f1_score(yt, pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        costs.append(int(fp) * COST_FP + int(fn) * COST_FN)
    prod_thr_rows.append({
        "thr": round(float(t), 2),
        "prec_mean": round(float(np.mean(precs)), 4),
        "rec_mean": round(float(np.mean(recs)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
        "cost_mean": int(np.mean(costs)),
    })

# Pick threshold by min prod cost (tie -> higher precision)
best = min(prod_thr_rows, key=lambda r: (r["cost_mean"], -r["prec_mean"]))
BEST_THR = float(best["thr"])

# Single prod sample for confusion example at BEST_THR
fi = rng.choice(fraud_idx, size=n_fraud, replace=False)
li = rng.choice(legit_idx, size=n_legit, replace=False)
idx = np.concatenate([fi, li])
yt = y_v.values[idx]
example = metrics_at(yt, proba[idx], BEST_THR)

out = {
    "lab": {"n": int(len(y_test)), "fraud_rate": 0.5,
            "f1_at_05": round(float(lab_f1), 4),
            "roc_auc": round(float(lab_roc), 4),
            "pr_auc": round(float(lab_pr), 4)},
    "prod_sim": {"n": N_PROD, "fraud_rate": PROD_FRAUD_RATE, "repeats": 5,
                 "cost_fp": COST_FP, "cost_fn": COST_FN,
                 "threshold_table": prod_thr_rows,
                 "example_at_best": example},
    "lab_threshold_table": lab_rows,
    "best_threshold": BEST_THR,
    "top_features": ["V15", "V11", "V5"],
    "note": "Balanced lab is trivially separable (~0.999). Prod sim shows precision collapse at 0.1% fraud. Threshold chosen by min expected cost on prod sim.",
}
with open("artifacts/prod_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
with open("artifacts/threshold.json", "w") as f:
    json.dump({"threshold": BEST_THR, "cost_fp": COST_FP, "cost_fn": COST_FN}, f, indent=2)
pd.DataFrame(prod_thr_rows).to_csv("artifacts/threshold_cost_table.csv", index=False)
print(f"BEST_THR={BEST_THR} example_prod={example} lab_f1={round(float(lab_f1),4)}")
