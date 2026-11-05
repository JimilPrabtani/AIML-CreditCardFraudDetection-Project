"""Fraud Triage Console (local) - loads frozen pipeline, never trains."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path("artifacts")
EXPECTED = [f"V{i}" for i in range(1, 29)] + ["Amount"]

@st.cache_resource
def load_artifacts():
    try:
        pipe = joblib.load(ART / "fraud_pipeline.pkl")
        thr = json.loads((ART / "threshold.json").read_text())
        prod = json.loads((ART / "prod_metrics.json").read_text()) if (ART / "prod_metrics.json").exists() else {}
        sample = pd.read_csv(ART / "sample_test_rows.csv")
    except FileNotFoundError as e:
        st.error(f"Missing artifact: {e.filename}. Run the notebook export cell, then `tasks/fix_honesty.py`.")
        st.stop()
    return pipe, float(thr.get("threshold", 0.5)), prod, sample

def validate_frame(df: pd.DataFrame):
    df = df.copy()
    dropped = [c for c in ["id", "Class", "pred", "proba"] if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
    for c in EXPECTED:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    missing = [c for c in EXPECTED if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED]
    return df, missing, extra, dropped

def risk_band(p: float, thr: float) -> str:
    if p >= thr:
        return "BLOCK — likely fraud"
    if p >= 0.5:
        return "REVIEW — borderline"
    return "LOW — likely legit"

pipe, THR, PROD, SAMPLE = load_artifacts()

st.set_page_config(page_title="Fraud Triage Console (local)", layout="wide")
st.title("Fraud Triage Console (local)")
st.caption(f"Frozen model: RandomForest (200 trees) | threshold={THR} (cost-based) | local only, no logging, no retrain")

tab1, tab2, tab3 = st.tabs(["Predict", "Performance", "About"])

with tab1:
    st.subheader("Single transaction (library sample + Amount what-if)")
    idx = st.selectbox("Pick a held-out sample row", options=list(SAMPLE.index),
                       format_func=lambda i: f"row {i} — Amount {SAMPLE.loc[i, 'Amount']:.2f}")
    base = SAMPLE.loc[idx, EXPECTED].copy()
    amt = st.slider("Amount (what-if, V1-V28 unchanged)", min_value=0.0,
                    max_value=float(max(25000.0, base['Amount'] * 1.5)),
                    value=float(base["Amount"]))
    row = base.copy()
    row["Amount"] = amt
    proba = float(pipe.predict_proba(pd.DataFrame([row]))[0, 1])
    dec = int(proba >= THR)
    c1, c2, c3 = st.columns(3)
    c1.metric("P(fraud)", f"{proba:.4f}")
    c2.metric("Decision", "FRAUD" if dec else "LEGIT")
    c3.metric("Risk", risk_band(proba, THR))
    st.progress(min(1.0, proba))
    st.info(f"Distance to threshold: {proba - THR:+.4f}. Global top features: V15, V11, V5 (RF importance). "
            f"V1-V28 are PCA-anonymized: no business meaning, Amount is the only interpretable knob.")
    st.divider()
    st.subheader("Batch CSV scoring")
    st.code("Expected columns: V1..V28, Amount  (id/Class are dropped/ignored)", language="text")
    up = st.file_uploader("Upload transactions CSV", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
        df, missing, extra, dropped = validate_frame(df_raw)
        if missing:
            st.error(f"Missing columns: {missing}")
        elif df[EXPECTED].isna().any().any():
            bad = df[EXPECTED].isna().sum()
            st.error(f"Non-numeric values in: {list(bad[bad > 0].index)}. Fix the CSV and retry.")
        else:
            if dropped:
                st.warning(f"Dropped non-feature columns: {dropped}" + (" (Class labels are never used for scoring.)" if "Class" in dropped else ""))
            if extra:
                st.warning(f"Ignored extra columns: {extra}")
            df = df[EXPECTED]
            with st.spinner("Scoring..."):
                p = pipe.predict_proba(df)[:, 1]
                out = df_raw.copy()
                out["proba"] = np.round(p, 4)
                out["decision"] = (p >= THR).astype(int)
                out["risk"] = [risk_band(x, THR) for x in p]
            st.write(f"Flagged: {(out['decision']==1).sum()} / {len(out)} at threshold {THR}")
            st.dataframe(out.head(50))
            st.download_button("Download scored CSV", out.to_csv(index=False).encode(),
                               file_name="scored_transactions.csv", mime="text/csv")

with tab2:
    st.subheader("Lab vs production reality")
    st.warning("Lab test is 50/50 balanced (F1 ~0.999). Real production is ~0.1% fraud: precision collapses. "
               "Threshold was chosen by expected cost on 50k-row 0.1% simulation, not by lab F1.")
    if PROD:
        c1, c2, c3 = st.columns(3)
        c1.metric("Lab F1 @0.5", PROD["lab"]["f1_at_05"])
        c2.metric("Lab ROC-AUC", PROD["lab"]["roc_auc"])
        c3.metric("Lab PR-AUC", PROD["lab"]["pr_auc"])
        st.write(f"Prod simulation: n={PROD['prod_sim']['n']}, fraud_rate={PROD['prod_sim']['fraud_rate']}, "
                 f"cost FP=${PROD['prod_sim']['cost_fp']} FN=${PROD['prod_sim']['cost_fn']}, best_thr={PROD['best_threshold']}")
        st.dataframe(pd.DataFrame(PROD["prod_sim"]["threshold_table"]).set_index("thr"))
        ex = PROD["prod_sim"]["example_at_best"]
        st.write(f"Example at thr={ex['thr']} on 50k prod-sim: TP={ex['tp']} FP={ex['fp']} FN={ex['fn']} TN={ex['tn']} "
                 f"prec={ex['prec']} rec={ex['rec']}")
        st.write("Lab sweep (balanced test, same-distribution):")
        st.dataframe(pd.DataFrame(PROD["lab_threshold_table"]).set_index("thr"))
    else:
        st.error("artifacts/prod_metrics.json missing. Run tasks/fix_honesty.py")

with tab3:
    st.subheader("Model card + limits")
    st.markdown("""
**Data:** `creditcard_2023.csv` (568,630 rows, 2023 snapshot). `V1-V28` PCA-anonymized, `Amount` only interpretable col. `id` dropped.
**Model:** sklearn `Pipeline(StandardScaler(Amount) + RandomForest 200 trees, max_depth 25)`. Frozen in `artifacts/fraud_pipeline.pkl`. App never retrains.
**Threshold:** cost-based on prod simulation (see Performance). Display value loaded from `artifacts/threshold.json`.
**Intended use:** SOC triage demo (batch review queue). Not a bank production system.
**Limits:** single 2023 snapshot (drift certain); balanced lab unrealistic; PCA blocks case explanation; no PII handling beyond in-memory scoring (nothing logged, nothing uploaded).
**Run:** `./.venv/Scripts/python.exe -m streamlit run app.py`
""")
