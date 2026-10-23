"""
Step 3: Build, train and test the Bayesian Network (expert-designed structure).

Run from the project root:  python train_model.py   (after preprocess.py)

Outputs:
  model/bn_model.pkl          trained model (refit on all data) + metadata, used by the Streamlit app
  model/metrics.json          test-set metrics
  model/confusion_matrix.png  confusion matrix on the test set
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
from pgmpy.inference import VariableElimination

# pgmpy changed its API between versions; support both (requirements.txt pins the tested one)
try:
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:                                   # older pgmpy
    from pgmpy.models import BayesianNetwork as DiscreteBayesianNetwork
try:
    from pgmpy.parameter_estimator import DiscreteBayesianEstimator as _Estimator   # pgmpy >= 1.0
    NEW_API = True
except ImportError:
    from pgmpy.estimators import BayesianEstimator as _Estimator                    # pgmpy 0.1.x
    NEW_API = False

DATA = "data/landslide_clean.csv"
MODEL_PATH = "model/bn_model.pkl"
METRICS_PATH = "model/metrics.json"
CM_PATH = "model/confusion_matrix.png"

TARGET = "severity"
CLASSES = ["low", "medium", "high"]
# 'accuracy' (location accuracy) is a data-quality attribute, not a landslide property a user can
# know in advance, so it is left out of the network.
FEATURES = ["season", "zone", "trigger", "category", "size", "setting"]

# Expert-designed structure (cause -> effect)
EDGES = [
    ("zone", "trigger"),      # climate/terrain of the region shapes which triggers occur
    ("season", "trigger"),    # monsoon season -> rain-type triggers
    ("zone", "category"),     # terrain decides the type of movement (rock fall vs landslide)
    ("zone", "setting"),      # road-cut slopes vs natural slopes differ by region
    ("trigger", "size"),      # heavy triggers produce bigger failures
    ("category", "size"),
    ("size", TARGET),         # bigger failures -> more fatalities
    ("trigger", TARGET),
    ("setting", TARGET),      # e.g. slopes above roads put people in the path
]
ESS = 10          # equivalent sample size for the BDeu prior (smoothing of sparse CPTs)
SEED = 42


def build_and_fit(train_df, state_names):
    model = DiscreteBayesianNetwork(EDGES)
    if NEW_API:
        est = _Estimator(state_names=state_names, prior_type="BDeu", equivalent_sample_size=ESS)
        model.fit(train_df, estimator=est)
    else:
        model.fit(train_df, estimator=_Estimator, prior_type="BDeu",
                  equivalent_sample_size=ESS, state_names=state_names)
    assert model.check_model()
    return model


def predict_proba(model, df, evidence_cols):
    """P(severity | evidence) for every row, using Variable Elimination."""
    infer = VariableElimination(model)
    out = []
    for _, row in df.iterrows():
        ev = {c: row[c] for c in evidence_cols}
        q = infer.query([TARGET], evidence=ev, show_progress=False)
        probs = dict(zip(q.state_names[TARGET], q.values))
        out.append([probs[c] for c in CLASSES])
    return np.array(out)


def decide(proba, priors=None):
    """Pick a class. priors=None -> plain MAP (most probable class).
    With priors -> divide by the class prior first, so rarer classes (medium/high) are not
    drowned out by 'low'. Better suited to risk assessment, where missing a high-severity
    event costs more than a false alarm."""
    score = proba if priors is None else proba / priors
    return np.array(CLASSES)[score.argmax(axis=1)]


def evaluate(y_true, proba, label, priors=None):
    y_pred = decide(proba, priors)
    order_idx = [CLASSES.index(c) for c in sorted(CLASSES)]   # sklearn needs sorted labels for AUC
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES,
                                                 average="macro", zero_division=0)
    res = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(p, 4), "macro_recall": round(r, 4), "macro_f1": round(f, 4),
        "roc_auc_ovr_macro": round(roc_auc_score(y_true, proba[:, order_idx], multi_class="ovr",
                                                 labels=sorted(CLASSES)), 4),
    }
    print(f"\n=== {label} ===")
    print(json.dumps(res, indent=2))
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    return res, y_pred


def main():
    df = pd.read_csv(DATA)
    df = df[FEATURES + [TARGET]].astype(str)
    state_names = {c: sorted(df[c].unique()) for c in df.columns}
    state_names[TARGET] = CLASSES

    # ---- train / test split (stratified so rare 'high' class appears in both) ----
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    print(f"Train: {len(train)}  Test: {len(test)}")
    majority = (test[TARGET] == train[TARGET].mode()[0]).mean()
    print(f"Majority-class baseline accuracy: {majority:.4f}")

    # ---- fit ----
    model = build_and_fit(train, state_names)
    print("\nNetwork edges:", list(model.edges()))

    # ---- evaluate ----
    priors = train[TARGET].value_counts(normalize=True).reindex(CLASSES).values
    proba_full = predict_proba(model, test, FEATURES)
    res_map, pred_map = evaluate(test[TARGET].values, proba_full,
                                 "Test set - all features, standard (MAP) decision")
    res_bal, pred_bal = evaluate(test[TARGET].values, proba_full,
                                 "Test set - all features, prior-adjusted (balanced) decision", priors)

    no_size = [c for c in FEATURES if c != "size"]          # missing-data demo
    proba_ns = predict_proba(model, test, no_size)
    res_ns, _ = evaluate(test[TARGET].values, proba_ns,
                         "Test set - 'size' unknown, prior-adjusted decision", priors)

    # ---- confusion matrices ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pred, title in [(axes[0], pred_map, "Standard (MAP)"), (axes[1], pred_bal, "Prior-adjusted")]:
        cm = confusion_matrix(test[TARGET], pred, labels=CLASSES)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    plt.suptitle("Bayesian Network - Confusion Matrices (test set)")
    plt.tight_layout(); plt.savefig(CM_PATH, dpi=150); plt.close()
    cm_bal = confusion_matrix(test[TARGET], pred_bal, labels=CLASSES)

    # ---- example inference queries (useful for the report) ----
    infer = VariableElimination(model)
    print("\n=== Example queries ===")
    q1 = infer.query([TARGET], evidence={"trigger": "downpour", "season": "monsoon", "size": "large_plus"}, show_progress=False)
    print("Forward  P(severity | downpour, monsoon, large_plus):\n", q1)
    q2 = infer.query(["trigger"], evidence={TARGET: "high"}, show_progress=False)
    print("Diagnostic  P(trigger | severity = high):\n", q2)
    q3 = infer.query([TARGET], evidence={"zone": "western_himalaya", "setting": "above_road"}, show_progress=False)
    print("Partial evidence  P(severity | western_himalaya, above_road):\n", q3)

    # ---- save metrics ----
    metrics = {"n_train": len(train), "n_test": len(test), "majority_baseline_accuracy": round(majority, 4),
               "all_features_map": res_map, "all_features_balanced": res_bal,
               "size_unknown_balanced": res_ns,
               "confusion_matrix_balanced": cm_bal.tolist(), "classes": CLASSES}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- refit on ALL data for the deployed app ----
    final = build_and_fit(df, state_names)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": final, "features": FEATURES, "target": TARGET,
                     "classes": CLASSES, "state_names": state_names,
                     "priors": dict(zip(CLASSES, priors))}, f)
    print(f"\nSaved {MODEL_PATH}, {METRICS_PATH}, {CM_PATH}")


if __name__ == "__main__":
    main()
