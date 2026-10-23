"""
Step 4: 5-fold cross-validation, baselines, and learned-vs-expert network structure.

Run from the project root:  python evaluate.py   (after preprocess.py; train_model.py must be in the same folder)

Outputs:
  model/cv_results.csv          mean and std of each metric per model
  model/cv_comparison.png       bar chart comparing the models
  model/network_expert.png      expert-designed structure
  model/network_learned.png     structure learned from data (hill climbing + BIC)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score)

from train_model import (DATA, FEATURES, TARGET, CLASSES, EDGES, ESS, SEED, NEW_API,
                         DiscreteBayesianNetwork, _Estimator, predict_proba, decide)

try:                                                     # pgmpy >= 1.0 (scikit-learn style API)
    from pgmpy.causal_discovery import HillClimbSearch
    NEW_HC = True
except ImportError:                                      # pgmpy 0.1.x
    from pgmpy.estimators import HillClimbSearch
    NEW_HC = False

COLS = FEATURES + [TARGET]


# ---------- Bayesian network helpers ----------
def fit_bn(edges, df, state_names):
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(COLS)
    model.add_edges_from(edges)
    if NEW_API:
        model.fit(df, estimator=_Estimator(state_names=state_names, prior_type="BDeu",
                                           equivalent_sample_size=ESS))
    else:
        model.fit(df, estimator=_Estimator, prior_type="BDeu",
                  equivalent_sample_size=ESS, state_names=state_names)
    return model


def learn_structure(df):
    if NEW_HC:
        hc = HillClimbSearch(scoring_method="bic-d", max_indegree=3, return_type="dag",
                             show_progress=False).fit(df[COLS])
        return list(hc.causal_graph_.edges())
    dag = HillClimbSearch(df[COLS]).estimate(scoring_method="bicscore", max_indegree=3, show_progress=False)
    return list(dag.edges())


def draw_network(edges, title, path):
    g = nx.DiGraph()
    g.add_nodes_from(COLS)
    g.add_edges_from(edges)
    colors = ["#e4572e" if n == TARGET else "#4c8bf5" for n in g.nodes()]
    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(g, seed=3, k=1.2)
    nx.draw_networkx(g, pos, node_color=colors, node_size=2600, font_size=8, font_color="white",
                     arrows=True, arrowsize=18, edge_color="#555")
    plt.title(title); plt.axis("off"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ---------- model wrappers: each returns P(class) in CLASSES order ----------
def reorder(proba, classes_):
    idx = [list(classes_).index(c) for c in CLASSES]
    return proba[:, idx]


def run_majority(train, test, priors, **_):
    return np.tile(priors, (len(test), 1))


def run_naive_bayes(train, test, state_names, **_):
    enc = OrdinalEncoder(categories=[state_names[c] for c in FEATURES])
    Xtr, Xte = enc.fit_transform(train[FEATURES]), enc.transform(test[FEATURES])
    nb = CategoricalNB(alpha=1.0, min_categories=[len(state_names[c]) for c in FEATURES])
    nb.fit(Xtr, train[TARGET])
    return reorder(nb.predict_proba(Xte), nb.classes_)


def run_logreg(train, test, **_):
    enc = OneHotEncoder(handle_unknown="ignore")
    Xtr, Xte = enc.fit_transform(train[FEATURES]), enc.transform(test[FEATURES])
    lr = LogisticRegression(max_iter=2000).fit(Xtr, train[TARGET])
    return reorder(lr.predict_proba(Xte), lr.classes_)


def run_bn_expert(train, test, state_names, **_):
    return predict_proba(fit_bn(EDGES, train, state_names), test, FEATURES)


def run_bn_learned(train, test, state_names, **_):
    return predict_proba(fit_bn(learn_structure(train), train, state_names), test, FEATURES)


MODELS = {
    "Majority baseline": run_majority,
    "Naive Bayes": run_naive_bayes,
    "Logistic Regression": run_logreg,
    "BN (expert structure)": run_bn_expert,
    "BN (learned structure)": run_bn_learned,
}


def score(y_true, proba, priors, balanced):
    y_pred = decide(proba, priors if balanced else None)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    _, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, average=None, zero_division=0)
    order = sorted(CLASSES)
    auc = roc_auc_score(y_true, proba[:, [CLASSES.index(c) for c in order]], multi_class="ovr", labels=order)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r,
            "macro_f1": f, "recall_high": rec[CLASSES.index("high")], "roc_auc": auc}


def main():
    df = pd.read_csv(DATA)[COLS].astype(str)
    state_names = {c: sorted(df[c].unique()) for c in df.columns}
    state_names[TARGET] = CLASSES

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    records = []
    for fold, (tr, te) in enumerate(skf.split(df, df[TARGET]), 1):
        train, test = df.iloc[tr], df.iloc[te]
        priors = train[TARGET].value_counts(normalize=True).reindex(CLASSES).values
        print(f"Fold {fold}/5 ...")
        for name, fn in MODELS.items():
            proba = fn(train=train, test=test, priors=priors, state_names=state_names)
            for balanced in (False, True):
                m = score(test[TARGET].values, proba, priors, balanced)
                records.append({"model": name, "decision": "balanced" if balanced else "standard",
                                "fold": fold, **m})

    res = pd.DataFrame(records)
    summary = (res.groupby(["model", "decision"])
                  [["accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_high", "roc_auc"]]
                  .agg(["mean", "std"]).round(3))
    order = list(MODELS)
    summary = summary.reindex(order, level=0)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\n===== 5-fold cross-validation (mean, std) =====")
    print(summary.to_string())
    summary.to_csv("model/cv_results.csv")

    # ---- comparison chart (balanced decision rule) ----
    bal = res[res.decision == "balanced"].groupby("model")[["macro_f1", "roc_auc", "recall_high"]].mean().reindex(order)
    ax = bal.plot(kind="bar", figsize=(9, 4.5), rot=20)
    ax.set_title("5-fold CV comparison (prior-adjusted decision)"); ax.set_ylabel("score")
    plt.tight_layout(); plt.savefig("model/cv_comparison.png", dpi=150); plt.close()

    # ---- structures: expert vs learned (on all data) ----
    learned = learn_structure(df)
    print("\nExpert edges :", EDGES)
    print("Learned edges:", learned)
    draw_network(EDGES, "Expert-designed structure", "model/network_expert.png")
    draw_network(learned, "Learned structure (hill climbing + BIC)", "model/network_learned.png")
    print("\nSaved model/cv_results.csv, cv_comparison.png, network_expert.png, network_learned.png")


if __name__ == "__main__":
    main()
