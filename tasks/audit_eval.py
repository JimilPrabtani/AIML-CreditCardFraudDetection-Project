"""Full audit eval: baselines + ablations + latency + prod-sim. Prints JSON for paper."""
import json, time, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, IsolationForest
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, accuracy_score)
from xgboost import XGBClassifier

RS = 42
df = pd.read_csv("creditcard_2023.csv")
X = df.drop(columns=["Class", "id"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
pre = ColumnTransformer([("amt", StandardScaler(), ["Amount"])], remainder="passthrough")

def ev(name, pipe, fit=True):
    t0 = time.perf_counter()
    if fit:
        pipe.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t1 = time.perf_counter()
    pr = pipe.predict_proba(X_test)[:, 1]
    yp = (pr >= 0.5).astype(int)
    # latency: single-row + batch-1000
    row = X_test.iloc[[0]]
    s = time.perf_counter(); pipe.predict_proba(row); single_ms = (time.perf_counter()-s)*1000
    batch = X_test.iloc[:1000]
    s = time.perf_counter(); pipe.predict_proba(batch); batch_ms = (time.perf_counter()-s)*1000
    tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0,1]).ravel()
    return {"model": name, "acc": round(float(accuracy_score(y_test, yp)),4),
            "prec": round(float(precision_score(y_test, yp)),4),
            "rec": round(float(recall_score(y_test, yp)),4),
            "f1": round(float(f1_score(y_test, yp)),4),
            "roc_auc": round(float(roc_auc_score(y_test, pr)),4),
            "pr_auc": round(float(average_precision_score(y_test, pr)),4),
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
            "fit_s": round(t_fit,1), "single_ms": round(single_ms,3), "batch1000_ms": round(batch_ms,1)}

results = []
results.append(ev("logreg", Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])))
results.append(ev("bagging", Pipeline([("pre", pre), ("clf", BaggingClassifier(random_state=RS))])))
# frozen RF (no refit) for shipped numbers
frozen = joblib.load("artifacts/fraud_pipeline.pkl")
results.append(ev("rf200_frozen", frozen, fit=False))
results.append(ev("xgb", Pipeline([("pre", pre), ("clf", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RS))])))
# isoforest (unsupervised contrast)
t0=time.perf_counter()
iso = IsolationForest(contamination=0.5, random_state=RS).fit(X_train)
t_fit=time.perf_counter()-t0
pred_iso = iso.predict(X_test)
yp_iso = np.where(pred_iso==-1,1,0)
tn,fp,fn,tp = confusion_matrix(y_test, yp_iso, labels=[0,1]).ravel()
results.append({"model":"isolationforest","acc":round(float(accuracy_score(y_test,yp_iso)),4),
  "prec":round(float(precision_score(y_test,yp_iso)),4),"rec":round(float(recall_score(y_test,yp_iso)),4),
  "f1":round(float(f1_score(y_test,yp_iso)),4),"roc_auc":None,"pr_auc":None,
  "fp":int(fp),"fn":int(fn),"tp":int(tp),"tn":int(tn),"fit_s":round(t_fit,1),"single_ms":None,"batch1000_ms":None})

# 5-fold CV on RF (fast check, depth25/200 trees -> use 3-fold to save time? use 5-fold on 50k subsample)
sub = X.sample(n=60000, random_state=RS); suby = y.loc[sub.index]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
rf_cv = Pipeline([("pre", ColumnTransformer([("amt", StandardScaler(), ["Amount"])], remainder="passthrough")),
                  ("clf", RandomForestClassifier(n_estimators=200, max_depth=25, n_jobs=-1, random_state=RS))])
scores = cross_validate(rf_cv, sub, suby, cv=cv, scoring=["f1","roc_auc","average_precision"], n_jobs=-1)
cv_out = {k: [round(float(x),4) for x in v] for k,v in {"f1":scores["test_f1"],"roc":scores["test_roc_auc"],"pr":scores["test_average_precision"]}.items()}
cv_out["mean_f1"] = round(float(np.mean(scores["test_f1"])),4)

# Ablations on frozen RF features: drop Amount / drop top-3 (V15,V11,V5) -> quick RF-50trees retrain for signal
def quick_abl(drop_cols, name):
    Xt = X_train.drop(columns=drop_cols); Xte = X_test.drop(columns=drop_cols)
    cols = [c for c in ["Amount"] if c not in drop_cols]
    tr = ColumnTransformer([("amt", StandardScaler(), cols)], remainder="passthrough") if cols else "passthrough"
    p = Pipeline([("pre", tr), ("clf", RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=RS))])
    p.fit(Xt, y_train); pr = p.predict_proba(Xte)[:,1]; yp=(pr>=0.5).astype(int)
    return {"ablation": name, "f1": round(float(f1_score(y_test, yp)),4),
            "roc_auc": round(float(roc_auc_score(y_test, pr)),4),
            "pr_auc": round(float(average_precision_score(y_test, pr)),4)}
abls = [quick_abl(["Amount"], "no_amount"),
        quick_abl(["V15","V11","V5"], "no_top3_V15_V11_V5"),
        quick_abl(["Amount","V15","V11","V5"], "no_amount_no_top3")]

# full RF-100 baseline for ablation reference
p = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=100, max_depth=25, n_jobs=-1, random_state=RS))])
p.fit(X_train, y_train); pr=p.predict_proba(X_test)[:,1]
full100 = {"ablation":"full_rf100_ref","f1":round(float(f1_score(y_test,(pr>=0.5).astype(int))),4),
           "roc_auc":round(float(roc_auc_score(y_test,pr)),4),"pr_auc":round(float(average_precision_score(y_test,pr)),4)}

out = {"baselines": results, "cv5_rf_60k": cv_out, "ablations": [full100]+abls,
       "n_test": int(len(y_test)), "note": "lab is 50/50 balanced; see prod_metrics.json for 0.1% sim"}
print(json.dumps(out, indent=2))
with open("artifacts/audit_metrics.json","w") as f: json.dump(out, f, indent=2)
