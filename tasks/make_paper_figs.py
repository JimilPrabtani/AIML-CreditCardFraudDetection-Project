"""Paper figures: confusion matrices, ROC/PR, cost-vs-threshold, importance. Uses frozen artifacts only (fast)."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

RS = 42
df = pd.read_csv("creditcard_2023.csv")
X = df.drop(columns=["Class", "id"]); y = df["Class"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
pipe = joblib.load("artifacts/fraud_pipeline.pkl")
proba = pipe.predict_proba(X_test)[:, 1]
prod = json.load(open("artifacts/prod_metrics.json"))

import os
os.makedirs("paper/figs", exist_ok=True)

# 1. Confusion matrix lab @0.5 and @0.85
for thr, name in [(0.5, "cm_lab_05"), (0.85, "cm_lab_085")]:
    cm = confusion_matrix(y_test, (proba >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred LEGIT", "Pred FRAUD"]); ax.set_yticklabels(["True LEGIT", "True FRAUD"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"RF confusion, lab 50/50 @ {thr}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"paper/figs/{name}.png", dpi=200); plt.close(fig)

# 2. ROC + PR (frozen RF)
fpr, tpr, _ = roc_curve(y_test, proba)
pc, rc, _ = precision_recall_curve(y_test, proba)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(fpr, tpr); ax[0].plot([0,1],[0,1],"--"); ax[0].set_title("ROC (lab)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
ax[1].plot(rc, pc); ax[1].set_title("Precision-Recall (lab)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
fig.tight_layout(); fig.savefig("paper/figs/roc_pr.png", dpi=200); plt.close(fig)

# 3. Cost vs threshold (prod sim)
t = pd.DataFrame(prod["prod_sim"]["threshold_table"])
fig, ax1 = plt.subplots(figsize=(7, 3.8))
ax1.plot(t["thr"], t["cost_mean"], marker="o", label="mean cost ($)")
ax1.set_xlabel("Threshold"); ax1.set_ylabel("Mean cost ($ / 50k txns)")
ax2 = ax1.twinx()
ax2.plot(t["thr"], t["prec_mean"], marker="s", color="green", label="precision")
ax2.plot(t["thr"], t["rec_mean"], marker="^", color="orange", label="recall")
ax2.set_ylabel("Precision / Recall")
ax1.set_title("Prod-sim (0.1% fraud): cost vs threshold")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig("paper/figs/cost_thr.png", dpi=200); plt.close(fig)

# 4. Feature importance (top 10)
imp = pipe.named_steps["clf"].feature_importances_
feats = list(X.columns)
o = np.argsort(imp)[::-1][:10]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.barh([feats[i] for i in o][::-1], imp[o][::-1])
ax.set_title("RF feature importance (top 10)")
ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig("paper/figs/importance.png", dpi=200); plt.close(fig)

print("figs done:", sorted(os.listdir("paper/figs")))
print("lab_cm05:", confusion_matrix(y_test, (proba>=0.5).astype(int), labels=[0,1]).ravel().tolist())
print("lab_cm085:", confusion_matrix(y_test, (proba>=0.85).astype(int), labels=[0,1]).ravel().tolist())
