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
import streamlit as st
from pgmpy.inference import VariableElimination

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "bn_model.pkl"

st.set_page_config(page_title="Landslide Risk - Bayesian Network", page_icon="⛰️", layout="wide")

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


try:
    bundle, infer = load_model()
except FileNotFoundError:
    st.error("Trained model not found. Run `python preprocess.py` and `python train_model.py` first.")
    st.stop()

FEATURES, TARGET, CLASSES = bundle["features"], bundle["target"], bundle["classes"]
STATES, PRIORS = bundle["state_names"], bundle["priors"]

st.title("⛰️ Landslide Risk Assessment in India")
st.caption("Bayesian Network inference trained on NASA's Global Landslide Catalog (India records)")

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
        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Severity probabilities")
            chart = pd.DataFrame({"probability": probs.reindex(CLASSES).values},
                                 index=[f"{c} ({SEVERITY_TEXT[c]})" for c in CLASSES])
            st.bar_chart(chart)
        with right:
            st.markdown("#### Result")
            msg = (f"**Risk level: {label.upper()}**  \n"
                   f"{SEVERITY_TEXT[label]} is the most elevated outcome compared with typical events.")
            {"low": st.success, "medium": st.warning, "high": st.error}[label](msg)
            for c in CLASSES:
                st.metric(f"P({c})", f"{probs[c]:.1%}", delta=f"typical {PRIORS[c]:.1%}", delta_color="off")
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
        st.bar_chart(out["risk_level"].value_counts().reindex(CLASSES).fillna(0))
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
        df = pd.DataFrame({"given your evidence": probs, "overall (no evidence)": base})
        df.index = [pretty(i) for i in df.index]
        st.bar_chart(df)
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
