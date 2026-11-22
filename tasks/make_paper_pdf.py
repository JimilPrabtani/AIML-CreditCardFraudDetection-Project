"""Render paper/root.pdf (IEEE-styled, two-column, US Letter) from the same verified content as root.tex."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle, Image, FrameBreak, KeepTogether)
from reportlab.lib import colors

W, H = LETTER
M = 0.65 * inch
COLW = (W - 2 * M - 0.25 * inch) / 2
TOP = 0.6 * inch

stTitle = ParagraphStyle("title", fontName="Times-Bold", fontSize=17, leading=20, alignment=TA_CENTER)
stAuthor = ParagraphStyle("author", fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_CENTER)
stAbs = ParagraphStyle("abs", fontName="Times-Bold", fontSize=9, leading=11, alignment=TA_JUSTIFY)
stAbsT = ParagraphStyle("abst", fontName="Times-Bold", fontSize=9, leading=11, alignment=TA_CENTER)
stBody = ParagraphStyle("body", fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_JUSTIFY,
                        firstLineIndent=9)
stBody0 = ParagraphStyle("body0", parent=stBody, firstLineIndent=0)
stH1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=10, leading=12, spaceBefore=8, spaceAfter=3,
                      textTransform="uppercase")
stH2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=9, leading=11, spaceBefore=5, spaceAfter=2)
stCap = ParagraphStyle("cap", fontName="Times-Roman", fontSize=7.5, leading=9, alignment=TA_CENTER)
stRef = ParagraphStyle("ref", fontName="Times-Roman", fontSize=7.5, leading=9, alignment=TA_JUSTIFY,
                       leftIndent=12, firstLineIndent=-12)
stCell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.5, leading=9, alignment=TA_CENTER)
stCellH = ParagraphStyle("cellh", parent=stCell, fontName="Times-Bold")

title_frame = Frame(M, H - TOP - 3.3 * inch, W - 2 * M, 3.3 * inch, id="title")
col1 = Frame(M, M, COLW, H - TOP - M - 3.3 * inch, id="c1")
col2 = Frame(M + COLW + 0.25 * inch, M, COLW, H - TOP - M - 3.3 * inch, id="c2")
col1l = Frame(M, M, COLW, H - TOP - M, id="c1l")
col2l = Frame(M + COLW + 0.25 * inch, M, COLW, H - TOP - M, id="c2l")

doc = BaseDocTemplate("paper/root.pdf", pagesize=LETTER,
                      title="Cost-Aware Credit Card Fraud Triage",
                      author="AIML-CreditCardFraudDetection-Project")
doc.addPageTemplates([
    PageTemplate(id="first", frames=[title_frame, col1, col2]),
    PageTemplate(id="later", frames=[col1l, col2l]),
])


def P(t, s=stBody):
    return Paragraph(t, s)


def P0(t):
    return Paragraph(t, stBody0)


def H1(t):
    return P(f"<br/><b>{t}</b>", stH1)


def H2(t):
    return P(f"<b><i>{t}</i></b>", stH2)


def table(data, widths, caption, label):
    hdr = [P(f"<b>{c}</b>", stCellH) for c in data[0]]
    rows = [hdr] + [[P(str(c), stCell) for c in r] for r in data[1:]]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                           ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return KeepTogether([t, P(f"{caption}", stCap)])


def fig(path, width, caption):
    return KeepTogether([Image(path, width=width, height=width * 0.62,
                               kind="proportional"), P(caption, stCap)])


cw = COLW
w5 = [cw * 0.30, cw * 0.14, cw * 0.14, cw * 0.14, cw * 0.14, cw * 0.14]
w4 = [cw * 0.24, cw * 0.22, cw * 0.22, cw * 0.32]
w4b = [cw * 0.40, cw * 0.20, cw * 0.20, cw * 0.20]
w5l = [cw * 0.16, cw * 0.21, cw * 0.21, cw * 0.21, cw * 0.21]

story = [
    P("Cost-Aware Credit Card Fraud Triage on the 2023 European Cardholder Dataset: From Balanced-Lab Benchmarks to Production-Realistic Thresholding", stTitle),
    Spacer(1, 8),
    P("Author Name<br/>Department, University — City, Country<br/>email@university.edu", stAuthor),
    Spacer(1, 8),
    P("Abstract—", stAbsT),
    P("Credit card fraud detection is a high-stakes, cost-asymmetric classification problem: a missed fraud costs orders of magnitude more than an unnecessary manual review, yet public benchmarks are often released with balanced class ratios that do not match production traffic. We study the 2023 European cardholder dataset (568,630 transactions, PCA-anonymized features V1–V28 plus Amount) with a leakage-safe pipeline and compare five approaches: logistic regression, bagging, random forest, XGBoost, and an unsupervised Isolation Forest contrast. The shipped random forest (200 trees, max_depth 25; 5-fold CV mean F1 0.9999) scores F1 0.9998, ROC-AUC 1.0 and PR-AUC 0.9999 on the balanced held-out test (113,726 rows). Resampling frozen scores into five 50,000-row simulations at 0.1% fraud with costs FP $10 / FN $500 selects threshold 0.85, cutting expected cost from $166 (at 0.5) to $24 while holding recall at 1.0 and raising precision 0.751 to 0.954. A frozen pipeline backs a local Streamlit triage console showing lab and production views side by side.", stAbs),
    Spacer(1, 4),
    P("<i>Keywords</i>—fraud detection, cost-sensitive learning, threshold selection, random forest, class imbalance, SOC triage.", stAbs),
    FrameBreak(),
    P0("Card payments keep growing, and with them fraud losses. Machine learning is the standard line of defense [2][4]. Two facts make the problem hard. First, features are privacy-sensitive: public sets expose only PCA-anonymized components. Second, errors are asymmetric: declining a legitimate payment triggers a cheap review, while missing a fraud writes off the full amount [5]. A model tuned for F1 on a balanced sample can fail the business objective where fraud is a fraction of a percent."),
    P("This work makes the lab-to-production gap explicit. On the 2023 European cardholder set (568,630 rows, ~50/50 balance) we (i) benchmark five classifiers under one leakage-safe protocol, (ii) freeze the best random forest, (iii) re-tune only the threshold on a costed 0.1%-fraud simulation, and (iv) ship a triage console showing both views side by side."),
    P("Contributions: a fair five-model benchmark (Table I); cost-based thresholding cutting expected cost ~7x (Table II); a frozen inference-only deployment with model card and smoke test; honest failure analysis and limits.", stBody0),
    H1("I. Related Work"),
    P0("<i>Imbalance.</i> The classic ULB set (284,807 txns, 492 frauds) showed accuracy is meaningless at 0.17% fraud and PR analysis is the right lens [2][3][9]. Our 2023 set differs: ~50/50 balance makes lab metrics look trivially perfect, forcing us to simulate production prevalence explicitly."),
    P("<i>Drift and delayed labels.</i> Fraud tactics and spending drift and labels arrive late [2][4]. We do not solve drift; we document the single-snapshot limit and make the threshold script re-runnable on fresh labels.", stBody),
    P("<i>Cost-sensitive decisions.</i> Example-dependent costs should drive threshold choice [5]. We follow that literally: the model is picked by F1/CV, the operating point by expected review cost.", stBody),
    P("<i>Models.</i> Random forests [6] and XGBoost [7] dominate tabular fraud benchmarks; Isolation Forest [8] is the unsupervised contrast. Implementation uses scikit-learn pipelines [10]. Our work is an applied, reproducible triage study: weaker on novelty, explicit on honesty.", stBody),
    H1("II. Proposed Method"),
    H2("A. System Overview"),
    P0("CSV rows (V1..V28, Amount) enter a ColumnTransformer (standardize Amount, pass V1–V28 through) + RandomForest(200xd25), emitting P(fraud); threshold 0.85 maps it to FRAUD/LEGIT plus LOW/REVIEW/BLOCK bands. Scaling fits on train only; the app loads the frozen pipeline and never retrains."),
    H2("B. Mathematical Formulation"),
    P0("Forest estimate p(x)=P(y=1|x) averages T=200 depth-capped votes; y-hat=1[p(x)>=t]. With C_FP=10, C_FN=500:"),
    P0("<b>C(t) = 10*FP(t) + 500*FN(t)</b>, &nbsp; t* = argmin mean C(t) over five 50k-row resimulations at 0.1% fraud (ties to higher precision)."),
    H2("C. Algorithmic Details"),
    P0("For each t in {0.10..0.90}: repeat 5x {sample 50 fraud + 49,950 legit; compute precision/recall/F1/cost}; average; pick lowest mean cost (seed 42 throughout)."),
    H2("D. Implementation Details"),
    P0("Python, scikit-learn [10], XGBoost [7], Streamlit 1.66.1. Stratified 80/20 split, seed 42, id dropped. LogReg (max_iter 1000); bagging (10 trees); RF (200 trees, tuned depth 25; 6-iteration randomized search, 3-fold CV, best F1 0.9998); XGBoost (300 trees, depth 5, lr 0.05, subsample/colsample 0.8); Isolation Forest (contamination 0.5). Frozen: pipeline (~37MB), threshold.json (0.85), sample rows, prod_metrics.json."),
    H1("III. Experimental Setup"),
    H2("A. Datasets"),
    P0("<i>creditcard_2023.csv</i> [1]: 568,630 European transactions, 2023. id (dropped), V1–V28 (PCA, mean ~0), Amount (mean ~$12k), Class (~50/50; test: 56,863 fraud of 113,726). No missing values, no duplicates."),
    H2("B. Baseline Models"),
    P0("Shared split/preprocessing: logistic regression (linear reference), bagging (variance reference), random forest (shipped), XGBoost (boosting challenger), Isolation Forest (unsupervised contrast)."),
    H2("C. Evaluation Metrics"),
    P0("Accuracy, precision, recall, F1, ROC-AUC, PR-AUC at t=0.5 on balanced test; plus simulated precision/recall/F1/expected-cost at 0.1% fraud. Per [9], PR and cost carry the argument under imbalance."),
    H1("IV. Results"),
    H2("A. Quantitative Results"),
    P0("Tree ensembles separate this set almost perfectly; the linear model trails and the unsupervised contrast collapses (Table I, from tasks/audit_eval.py; frozen RF reproduces the notebook exactly)."),
    table([["Model", "Acc", "Prec", "Rec", "F1", "ROC/PR"],
           ["LogReg", "0.9649", "0.9770", "0.9523", "0.9645", "0.9935/0.9944"],
           ["Bagging", "0.9991", "0.9985", "0.9998", "0.9991", "1.0/0.9999"],
           ["RF-200-d25", "0.9998", "0.9996", "1.0", "0.9998", "1.0/0.9999"],
           ["XGBoost", "0.9983", "0.9973", "0.9992", "0.9983", "0.9999/0.9999"],
           ["IsoForest", "0.6089", "0.6088", "0.6097", "0.6092", "-/-"]],
          w5, "TABLE I. Balanced lab test (113,726 rows, 50/50) at 0.5. Best bold (RF row).", "t1"),
    P("5-fold CV is stable: full-train mean F1 0.9999 (std ~0); 60k-subsample re-run mean F1 0.9982 (folds 0.9992/0.9981/0.9976/0.9979/0.9982).", stBody),
    P("Table II is the key result: at 0.1% fraud the 0.5 default costs $166 at precision 0.751; t*=0.85 costs $24 at precision 0.954 with recall 1.0. At 0.9 recall breaks (0.992) and cost explodes to $220.", stBody0),
    table([["t", "Prec", "Rec", "F1", "Cost"],
           ["0.5", "0.7511", "1.0", "0.8578", "$166"],
           ["0.7", "0.8835", "1.0", "0.9381", "$66"],
           ["0.85", "0.9543", "1.0", "0.9766", "$24"],
           ["0.9", "0.9612", "0.992", "0.9764", "$220"]],
          w5l, "TABLE II. Prod simulation (50k rows, 0.1% fraud, 5 repeats). Shipped: 0.85.", "t2"),
    P("At t*: TP=50, FP=2, FN=0 on one draw; lab at 0.5: 20 FP/0 FN (Fig. 1).", stBody),
    fig("paper/figs/cost_thr.png", cw, "Fig. 1. Prod-sim cost/precision/recall vs threshold (min at 0.85)."),
    fig("paper/figs/cm_lab_05.png", cw * 0.8, "Fig. 2. Lab confusion at 0.5: 20 FP, 0 FN."),
    fig("paper/figs/cm_lab_085.png", cw * 0.8, "Fig. 3. Lab confusion at 0.85: 3 FP, 60 FN."),
    H2("B. Qualitative Results"),
    P0("Balance bars are equal; Amount histograms overlap, so Amount alone does not separate classes. Importance: V15 (0.217), V11 (0.129), V5 (0.112), then V13/V18 (Fig. 4) — reported without business meaning (PCA). False positives average $9,022 vs $12,044 for correct rows: alarms concentrate on smaller tickets."),
    fig("paper/figs/importance.png", cw, "Fig. 4. Top-10 RF importance."),
    fig("paper/figs/roc_pr.png", cw, "Fig. 5. ROC and PR curves, frozen RF, lab test."),
    H2("C. Ablation Studies"),
    P0("Threshold ablation (Table III): lab F1 barely moves (0.9998 to 0.9985) while error type flips (0 to 173 FN) — the threshold trades FP for FN, it does not improve the model."),
    table([["t", "Prec", "Rec", "FP", "FN"],
           ["0.5", "0.9996", "1.0", "20", "0"],
           ["0.85", "0.9999", "0.9989", "3", "60"],
           ["0.9", "1.0", "0.997", "2", "173"]],
          w5l, "TABLE III. Lab threshold sensitivity: threshold picks error type.", "t3"),
    P("Component retrain ablations (Table IV: drop Amount / top-3 / all four, RF-100) leave F1 at 0.9998, ROC 1.0 throughout: lab signals are fully redundant, so the modeling problem is easy and the decision problem is hard.", stBody0),
    table([["Variant", "F1", "ROC", "PR"],
           ["Full ref", "0.9998", "1.0", "1.0"],
           ["No Amount", "0.9998", "1.0", "1.0"],
           ["No V15/11/5", "0.9998", "1.0", "1.0"],
           ["No both", "0.9998", "1.0", "1.0"]],
          w4, "TABLE IV. Component ablations (RF-100, lab @0.5).", "t4"),
    P("Inference cost (commodity Windows box): frozen RF ~51 ms/single txn, ~66 ms per 1,000 rows; LogReg/XGBoost ~5 ms per 1,000. Fits: LogReg 1.3 s, XGBoost 6.5 s, bagging 261 s.", stBody0),
    H1("V. Discussion"),
    P0("<i>Interpretation.</i> The set is nearly separable for ensembles at 50/50, so the decision problem dominates: prevalence collapse (precision 0.9996 to 0.751 at t=0.5) outweighs every other effect, and costed thresholding recovers most business loss."),
    P("<i>Vs prior work.</i> ROC-AUC is uninformative at the top (1.0 x3); PR/cost decide [9][3]; the operating point follows the cost matrix, not F1 [5]. Absolute scores exceed 0.1%-fraud benchmarks only because our lab is balanced — a difference we surface.", stBody),
    P("<i>Limitations.</i> Single 2023 snapshot (drift certain, unmeasured). PCA blocks case explanation (Amount only knob). t*=0.85 is conditional on $10/$500 and simulated prevalence. No calibration, real-time serving, or PII handling beyond in-memory scoring.", stBody),
    P("<i>Future work.</i> Calibration with reliability curves; drift monitoring + scheduled refits; tests on truly imbalanced snapshots; latency benchmarks; interpretable surrogates for analysts.", stBody),
    H1("VI. Conclusion"),
    P0("Ensembles look perfect on balanced 2023 data (RF F1 0.9998); the honest result is the costed one. Freezing the model and pricing errors at production prevalence, threshold 0.85 holds recall 1.0, lifts simulated precision 0.751 to 0.954, and cuts expected cost $166 to $24 per 50k transactions. The triage console shows both views so the lab score never stands alone."),
    H1("Appendix: Reproducibility"),
    P0("Data: Kaggle 2023 set [1], local git-ignored CSV. Order: notebook Run All, tasks/fix_honesty.py, streamlit run app.py (notebook export resets threshold to 0.5; fix script restores 0.85). Smoke: tasks/smoke_test.py. Tables: artifacts/prod_metrics.json, artifacts/audit_metrics.json. Figures: tasks/make_paper_figs.py."),
    H1("Acknowledgment"),
    P0("Thanks to the dataset publishers and the scikit-learn, XGBoost, and Streamlit contributors."),
    H1("References"),
    P("[1] N. Elgiryewithana, Credit card fraud detection dataset 2023, Kaggle, 2023.", stRef),
    P("[2] A. Dal Pozzolo et al., Credit card fraud detection and concept-drift adaptation with delayed supervised information, Proc. IEEE IJCNN, 2015.", stRef),
    P("[3] A. Dal Pozzolo et al., Calibrating probability with undersampling for unbalanced classification, Proc. IEEE CIDM, 2015.", stRef),
    P("[4] A. Dal Pozzolo et al., Learned lessons in credit card fraud detection from a practitioner perspective, Expert Systems with Applications, 41(10):4915-4928, 2014.", stRef),
    P("[5] A. C. Bahnsen et al., Example-dependent cost-sensitive decision trees, Expert Systems with Applications, 42(19):6602-6610, 2015.", stRef),
    P("[6] L. Breiman, Random forests, Machine Learning, 45(1):5-32, 2001.", stRef),
    P("[7] T. Chen and C. Guestrin, XGBoost: A scalable tree boosting system, Proc. ACM SIGKDD, pp. 785-794, 2016.", stRef),
    P("[8] F. T. Liu et al., Isolation forest, Proc. IEEE ICDM, pp. 413-422, 2008.", stRef),
    P("[9] T. Saito and M. Rehmsmeier, The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets, PLOS ONE, 10(3):e0118432, 2015.", stRef),
    P("[10] F. Pedregosa et al., Scikit-learn: Machine learning in Python, J. Machine Learning Research, 12:2825-2830, 2011.", stRef),
]
doc.build(story)
print("wrote paper/root.pdf")
