"""
Step 5/6: Streamlit app - Landslide Risk Assessment in India using Bayesian Network Inference.
Run locally:  streamlit run app.py
All outputs come from the trained model in model/bn_model.pkl (nothing is hard-coded).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pgmpy.inference import VariableElimination

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "bn_model.pkl"

st.set_page_config(page_title="Landslide Risk - Bayesian Network", page_icon="⛰️", layout="wide")

RISK_COLORS = {"low": "#2e9e6b", "medium": "#f0a30a", "high": "#d64545"}
TEAL, GREY = "#1f8a70", "#9aa7b2"

st.markdown("""
<style>
.hero {background: linear-gradient(100deg, #0f5c4d 0%, #1f8a70 55%, #e9c46a 130%);
       padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.02rem; color: white;}
.risk-card {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: .8rem;}
.risk-small {font-size: .85rem; letter-spacing: .06em; text-transform: uppercase; opacity: .75;}
.risk-big {font-size: 2.4rem; font-weight: 800; line-height: 1.15;}
.stat-card {padding: .8rem 1rem; border-radius: 12px; border-top: 5px solid; background: #f6f9f8;
            text-align: center;}
.stat-name {font-size: .85rem; opacity: .75;}
.stat-value {font-size: 1.7rem; font-weight: 700;}
.stat-typ {font-size: .75rem; opacity: .6;}
button[data-baseweb="tab"] {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

NOT_SPECIFIED = "— not specified —"
LABELS = {
    "season": "Season", "zone": "Region / zone", "trigger": "Trigger", "category": "Landslide type",
    "size": "Landslide size", "setting": "Setting", "severity": "Severity (fatalities)",
}
HELP = {
    "season": "Indian meteorological season of the event",
    "zone": "Approximate physiographic zone derived from coordinates",
    "trigger": "What set the landslide off",
    "category": "Type of mass movement",
    "size": "Reported size class",
    "setting": "Where the slope is (e.g. above a road)",
}
SEVERITY_TEXT = {
    "low": "No fatalities expected",
    "medium": "1 to 4 fatalities",
    "high": "5 or more fatalities",
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle, VariableElimination(bundle["model"])


def pretty(value):
    return str(value).replace("_", " ")


def query(infer, variable, evidence):
    ev = {k: v for k, v in evidence.items() if v and v != NOT_SPECIFIED and k != variable}
    q = infer.query([variable], evidence=ev, show_progress=False)
    return pd.Series(q.values, index=q.state_names[variable]), ev


def decide_label(probs, priors):
    """Prior-adjusted decision: compare each class to its everyday base rate."""
    pri = np.array([priors[c] for c in probs.index])
    return probs.index[int(np.argmax(probs.values / pri))]


def risk_card(label):
    c = RISK_COLORS[label]
    st.markdown(f"""<div class="risk-card" style="border-left: 9px solid {c}; background: {c}22;">
        <div class="risk-small">Predicted risk level</div>
        <div class="risk-big" style="color: {c};">{label.upper()}</div>
        <div>{SEVERITY_TEXT[label]} is the most elevated outcome compared with typical events.</div>
        </div>""", unsafe_allow_html=True)


def stat_card(name, value, typical, color):
    st.markdown(f"""<div class="stat-card" style="border-top-color: {color};">
        <div class="stat-name">P({name})</div><div class="stat-value" style="color: {color};">{value}</div>
        <div class="stat-typ">typical {typical}</div></div>""", unsafe_allow_html=True)


def prob_chart(probs):
    df = pd.DataFrame({"severity": CLASSES, "probability": [float(probs[c]) for c in CLASSES]})
    base = alt.Chart(df).encode(
        x=alt.X("severity:N", sort=CLASSES, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None))
    bars = base.mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=70).encode(
        color=alt.Color("severity:N", scale=alt.Scale(domain=CLASSES, range=[RISK_COLORS[c] for c in CLASSES]),
                        legend=None),
        tooltip=["severity", alt.Tooltip("probability:Q", format=".1%")])
    text = base.mark_text(dy=-8, fontSize=14, fontWeight="bold").encode(text=alt.Text("probability:Q", format=".0%"))
    return (bars + text).properties(height=300, width="container")


def compare_chart(probs, base_probs):
    rows = [{"state": pretty(k), "series": "given your evidence", "probability": float(v)} for k, v in probs.items()]
    rows += [{"state": pretty(k), "series": "overall (no evidence)", "probability": float(v)}
             for k, v in base_probs.items()]
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("state:N", title=None, axis=alt.Axis(labelAngle=-25)),
        xOffset="series:N",
        y=alt.Y("probability:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("series:N", scale=alt.Scale(domain=["given your evidence", "overall (no evidence)"],
                                                    range=[TEAL, GREY]), legend=alt.Legend(title=None, orient="top")),
        tooltip=["state", "series", alt.Tooltip("probability:Q", format=".1%")]
    ).properties(height=320, width="container")


def count_chart(counts):
    df = pd.DataFrame({"risk_level": CLASSES, "events": [int(counts.get(c, 0)) for c in CLASSES]})
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, size=70).encode(
        x=alt.X("risk_level:N", sort=CLASSES, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("events:Q", title="events", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("risk_level:N", scale=alt.Scale(domain=CLASSES, range=[RISK_COLORS[c] for c in CLASSES]),
                        legend=None)).properties(height=260, width="container")


try:
    bundle, infer = load_model()
except FileNotFoundError:
    st.error("Trained model not found. Run `python preprocess.py` and `python train_model.py` first.")
    st.stop()

FEATURES, TARGET, CLASSES = bundle["features"], bundle["target"], bundle["classes"]
STATES, PRIORS = bundle["state_names"], bundle["priors"]

st.markdown("""<div class="hero"><h1>⛰️ Landslide Risk Assessment in India</h1>
<p>Bayesian Network inference trained on NASA's Global Landslide Catalog (India records)</p></div>""",
            unsafe_allow_html=True)

tab_pred, tab_batch, tab_explore, tab_perf, tab_about = st.tabs(
    ["Risk prediction", "Batch (file upload)", "Explore the network", "Model performance", "About"])

# ---------------------------------------------------------------- 1. single prediction
with tab_pred:
    st.subheader("Describe the situation")
    st.write("Fill in what you know. Leave anything unknown as *not specified*; "
             "the network handles missing information.")
    cols = st.columns(3)
    evidence = {}
    for i, feat in enumerate(FEATURES):
        with cols[i % 3]:
            evidence[feat] = st.selectbox(LABELS[feat], [NOT_SPECIFIED] + STATES[feat],
                                          format_func=lambda v: v if v == NOT_SPECIFIED else pretty(v),
                                          help=HELP[feat], key=f"pred_{feat}")

    if st.button("Predict risk", type="primary"):
        probs, used = query(infer, TARGET, evidence)
        label = decide_label(probs, PRIORS)
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Severity probabilities")
            st.altair_chart(prob_chart(probs))
        with right:
            st.markdown("#### Result")
            risk_card(label)
            c1, c2, c3 = st.columns(3)
            for col, c in zip((c1, c2, c3), CLASSES):
                with col:
                    stat_card(c, f"{probs[c]:.0%}", f"{PRIORS[c]:.0%}", RISK_COLORS[c])
            st.caption(f"Based on {len(used)} known factor(s). Typical = share of this severity among all "
                       "historical events in the data.")

# ---------------------------------------------------------------- 2. batch upload
with tab_batch:
    st.subheader("Predict for many events at once")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in FEATURES) +
             ". Empty or unrecognised values are treated as unknown.")
    template = pd.DataFrame({c: [STATES[c][0], STATES[c][-1]] for c in FEATURES})
    st.download_button("Download a sample template", template.to_csv(index=False),
                       file_name="sample_input.csv", mime="text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
        elif st.button("Analyze file"):
            data = data.head(1000)
            rows = []
            bar = st.progress(0.0)
            for i, (_, r) in enumerate(data.iterrows()):
                ev = {c: str(r[c]) for c in FEATURES if str(r[c]) in STATES[c]}
                probs, _ = query(infer, TARGET, ev)
                rows.append({**{f"P({c})": round(float(probs[c]), 3) for c in CLASSES},
                             "risk_level": decide_label(probs, PRIORS)})
                bar.progress((i + 1) / len(data))
            st.session_state["batch_out"] = pd.concat(
                [data.reset_index(drop=True), pd.DataFrame(rows)], axis=1)   # kept so results survive reruns
    if "batch_out" in st.session_state:
        out = st.session_state["batch_out"]
        st.dataframe(out)
        st.altair_chart(count_chart(out["risk_level"].value_counts()))
        st.download_button("Download results", out.to_csv(index=False),
                           file_name="landslide_predictions.csv", mime="text/csv")
        st.caption("Only the first 1000 rows are processed.")

# ---------------------------------------------------------------- 3. explore (any direction)
with tab_explore:
    st.subheader("Ask the network any question")
    st.write("Pick something to infer and set what you know. Example: infer the **trigger** "
             "given that severity was **high**.")
    all_vars = FEATURES + [TARGET]
    target_var = st.selectbox("Infer this variable", all_vars, index=all_vars.index("trigger"),
                              format_func=lambda v: LABELS[v], key="exp_target")
    ecols = st.columns(3)
    ev2 = {}
    for i, v in enumerate([x for x in all_vars if x != target_var]):
        with ecols[i % 3]:
            opts = STATES[v] if v != TARGET else CLASSES
            ev2[v] = st.selectbox(LABELS[v], [NOT_SPECIFIED] + list(opts),
                                  format_func=lambda x: x if x == NOT_SPECIFIED else pretty(x),
                                  key=f"exp_{v}")
    if st.button("Run inference"):
        probs, used = query(infer, target_var, ev2)
        base, _ = query(infer, target_var, {})
        st.altair_chart(compare_chart(probs, base))
        df = pd.DataFrame({"given your evidence": probs, "overall (no evidence)": base})
        df.index = [pretty(i) for i in df.index]
        st.dataframe(df.style.format("{:.1%}"))
        st.caption("Comparing the two columns shows which outcomes the evidence makes more or less likely.")

# ---------------------------------------------------------------- 4. performance
with tab_perf:
    st.subheader("How well does the model work?")
    mpath = BASE / "model" / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.markdown(f"**Hold-out test** ({m['n_train']} train / {m['n_test']} test records). "
                    f"Always predicting the most common class scores "
                    f"{m['majority_baseline_accuracy']:.1%} accuracy.")
        rows = {"Standard decision": m["all_features_map"], "Prior-adjusted decision": m["all_features_balanced"],
                "Size unknown (prior-adjusted)": m["size_unknown_balanced"]}
        st.dataframe(pd.DataFrame(rows).T)
    cv = BASE / "model" / "cv_results.csv"
    if cv.exists():
        st.markdown("**5-fold cross-validation** (mean and standard deviation across folds)")
        st.dataframe(pd.read_csv(cv, header=[0, 1], index_col=[0, 1]))
    for img, cap in [("confusion_matrix.png", "Confusion matrices (test set)"),
                     ("cv_comparison.png", "Model comparison (5-fold CV)"),
                     ("network_expert.png", "Expert-designed network structure"),
                     ("network_learned.png", "Structure learned from data")]:
        p = BASE / "model" / img
        if p.exists():
            st.image(str(p), caption=cap)

# ---------------------------------------------------------------- 5. about
with tab_about:
    st.subheader("About this project")
    st.markdown("""
**What it does:** estimates how severe a landslide is likely to be (by fatalities) from the season, region,
trigger, type, size and setting, using a Bayesian Network built with `pgmpy`.

**Data:** NASA Global Landslide Catalog (one-time export, current to 2016), India records only.
Most events are rain-triggered and come from media reports.

**Limitations:** the catalog has no information on population exposure, time of day or terrain, which strongly
affect fatalities, so predictions are probabilities and not certainties. Zones are approximated from
coordinates. This is an academic prototype and **must not be used for real safety decisions**.
""")
