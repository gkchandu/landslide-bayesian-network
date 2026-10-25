"""
Tune the prior strength (equivalent sample size, ESS) of the Bayesian Network with 5-fold cross-validation.

Why: severity has three parents (size x trigger x setting = 96 combinations) but only ~1,000 records, so many
tables rest on 0-2 events. A weak prior lets a single event dominate; a stronger prior smooths the tables.

Run from the project root:  python tune_ess.py
Outputs: model/ess_tuning.csv, model/ess_tuning.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import evaluate as ev
from train_model import DATA, EDGES, CLASSES, TARGET, SEED, FEATURES, predict_proba

ESS_VALUES = [1, 10, 50, 100, 200, 500, 1000]


def log_loss(y_true, proba):
    idx = np.array([CLASSES.index(y) for y in y_true])
    p = np.clip(proba[np.arange(len(y_true)), idx], 1e-12, 1)
    return float(-np.mean(np.log(p)))


def main():
    df = pd.read_csv(DATA)[ev.COLS].astype(str)
    state_names = {c: sorted(df[c].unique()) for c in df.columns}
    state_names[TARGET] = CLASSES
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for ess in ESS_VALUES:
        ev.ESS = ess                                    # fit_bn in evaluate.py reads this value
        print(f"ESS = {ess} ...")
        for tr, te in skf.split(df, df[TARGET]):
            train, test = df.iloc[tr], df.iloc[te]
            priors = train[TARGET].value_counts(normalize=True).reindex(CLASSES).values
            proba = predict_proba(ev.fit_bn(EDGES, train, state_names), test, FEATURES)
            y = test[TARGET].values
            std, bal = ev.score(y, proba, priors, False), ev.score(y, proba, priors, True)
            rows.append({"ess": ess, "log_loss": log_loss(y, proba), "roc_auc": std["roc_auc"],
                         "acc_standard": std["accuracy"], "f1_standard": std["macro_f1"],
                         "acc_balanced": bal["accuracy"], "f1_balanced": bal["macro_f1"],
                         "recall_high_balanced": bal["recall_high"]})

    res = pd.DataFrame(rows).groupby("ess").mean().round(4)
    print("\n===== ESS tuning (5-fold CV means) =====")
    print(res.to_string())
    best = res["log_loss"].idxmin()
    print(f"\nBest ESS by log-loss: {best}")
    print("Set  ESS = %d  in train_model.py, then rerun train_model.py and evaluate.py." % best)
    res.to_csv("model/ess_tuning.csv")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(res.index, res["log_loss"], "o-", color="#d64545", label="log-loss (lower is better)")
    ax1.set_xscale("log"); ax1.set_xlabel("Equivalent sample size (prior strength)"); ax1.set_ylabel("log-loss")
    ax2 = ax1.twinx()
    ax2.plot(res.index, res["roc_auc"], "s--", color="#1f8a70", label="AUC (higher is better)")
    ax2.set_ylabel("AUC")
    ax1.axvline(best, color="grey", ls=":")
    fig.legend(loc="upper center", ncol=2, fontsize=8)
    plt.title("Tuning the prior strength (5-fold CV)", y=1.12)
    plt.tight_layout(); plt.savefig("model/ess_tuning.png", dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
