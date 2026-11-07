"""Fraud Triage Console (local) - loads frozen pipeline, never trains."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ART = Path("artifacts")
EXPECTED = [f"V{i}" for i in range(1, 29)] + ["Amount"]
COLORS = {"BLOCK": "#e5484d", "REVIEW": "#f5a524", "LOW": "#30a46c"}


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
        return "BLOCK"
    if p >= 0.5:
        return "REVIEW"
    return "LOW"


def pill(label: str) -> str:
    c = COLORS[label]
    return f"<span style='background:{c}22;color:{c};border:1px solid {c}55;padding:4px 14px;border-radius:999px;font-weight:700;font-size:0.85rem'>{label}</span>"


pipe, THR, PROD, SAMPLE = load_artifacts()

st.set_page_config(page_title="Fraud Triage", page_icon="🛡️", layout="wide")
st.markdown("<style>section.main > div{max-width:1100px} h1{letter-spacing:-0.02em}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛡️ Fraud Triage")
    st.caption("Local SOC demo · nothing stored")
    st.divider()
    st.metric("Threshold", f"{THR:.2f}")
    if PROD:
        st.caption(f"Cost FP ${PROD['prod_sim']['cost_fp']} · FN ${PROD['prod_sim']['cost_fn']}")
        st.caption(f"Prod sim: 0.1% fraud · Lab F1 {PROD['lab']['f1_at_05']}")
    st.caption("RandomForest · 200 trees")

st.title("Fraud Triage")
st.caption("Frozen model scores transactions as FRAUD / LEGIT. Lab is 50/50 balanced — real fraud is ~0.1%, so check the Performance tab.")

tab_score, tab_perf, tab_about = st.tabs(["Score", "Performance", "About"])

with tab_score:
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Single transaction**")
            idx = st.selectbox("Sample row", options=list(SAMPLE.index),
                               format_func=lambda i: f"row {i} · ${SAMPLE.loc[i, 'Amount']:,.2f}")
            base = SAMPLE.loc[idx, EXPECTED].copy()
            amt = st.slider("Amount (what-if)", 0.0, float(max(25000.0, base["Amount"] * 1.5)), float(base["Amount"]))
            row = base.copy()
            row["Amount"] = amt

    proba = float(pipe.predict_proba(pd.DataFrame([row]))[0, 1])
    label = risk_band(proba, THR)
    with right:
        with st.container(border=True):
            st.markdown("**Verdict**")
            st.markdown(f"# {proba:.4f} &nbsp; {pill(label)}", unsafe_allow_html=True)
            st.caption(f"{'FRAUD' if proba >= THR else 'LEGIT'} · {proba - THR:+.4f} from threshold · P(fraud)")
            st.progress(min(1.0, proba))
            st.caption("Top signals: V15 · V11 · V5 (V1–V28 are PCA-anonymized, Amount is the only knob).")

    with st.container(border=True):
        st.markdown("**Batch scoring**")
        st.caption("CSV with `V1–V28, Amount`. `id` / `Class` are dropped, never scored.")
        up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if up is not None:
            df_raw = pd.read_csv(up)
            df, missing, extra, dropped = validate_frame(df_raw)
            if missing:
                st.error(f"Missing columns: {missing}")
            elif df[EXPECTED].isna().any().any():
                bad = df[EXPECTED].isna().sum()
                st.error(f"Non-numeric values in: {list(bad[bad > 0].index)}")
            else:
                if dropped:
                    st.warning(f"Dropped: {dropped}")
                if extra:
                    st.warning(f"Ignored extras: {extra}")
                p = pipe.predict_proba(df[EXPECTED])[:, 1]
                out = df_raw.copy()
                out["proba"] = p.round(4)
                out["decision"] = (p >= THR).astype(int)
                out["risk"] = [risk_band(x, THR) for x in p]
                st.write(f"Flagged **{(out['decision'] == 1).sum()} / {len(out)}** at {THR}")
                st.dataframe(out.head(50), use_container_width=True)
                st.download_button("Download scored CSV", out.to_csv(index=False).encode(), "scored_transactions.csv", "text/csv")

with tab_perf:
    if not PROD:
        st.error("`artifacts/prod_metrics.json` missing. Run `tasks/fix_honesty.py`.")
    else:
        st.warning("Lab is 50/50 (F1 ~0.999). Production is ~0.1% fraud — precision collapses. Threshold picked by cost, not lab F1.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Lab F1 @0.5", PROD["lab"]["f1_at_05"])
        c2.metric("Lab ROC-AUC", PROD["lab"]["roc_auc"])
        c3.metric("Lab PR-AUC", PROD["lab"]["pr_auc"])
        st.markdown("**Prod simulation (50k rows, 0.1% fraud)**")
        st.dataframe(pd.DataFrame(PROD["prod_sim"]["threshold_table"]).set_index("thr"), use_container_width=True)
        ex = PROD["prod_sim"]["example_at_best"]
        st.caption(f"At {ex['thr']}: TP={ex['tp']} FP={ex['fp']} FN={ex['fn']} · prec={ex['prec']} rec={ex['rec']}")

with tab_about:
    with st.container(border=True):
        st.markdown("""**Data** — `creditcard_2023.csv`, 568k rows, 2023 snapshot. V1–V28 PCA-anonymized, `id` dropped.
**Model** — `Pipeline(StandardScaler(Amount) + RandomForest 200×d25)`, frozen in `artifacts/fraud_pipeline.pkl`. App never retrains.
**Threshold** — 0.85, cost-based (FP $10 / FN $500) on prod simulation. See Performance tab.
**Use** — SOC triage demo only. Not a production bank system.
**Limits** — 2023 snapshot (drift), balanced lab overstates precision, PCA blocks explanations, in-memory scoring only.""")
