"""Rail: reproduce the current model honestly, and quantify the speed confound.

Nothing here changes a prediction. It answers three questions:

  1. Does the reported 0.721 +/- 0.019 reproduce?
  2. How much of it is the speed shortcut? Every one of the 133 training files
     below 9.7 m/s is Normal, so "slow means Normal" is free accuracy.
  3. What is the honest nested-CV number, where the config is chosen inside
     the fold rather than on the folds it is scored on?
"""
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate as V

DATA = os.environ.get("NEBULAX_DATA", "data")
LABELS_DIR = os.environ.get("NEBULAX_LABELS", os.path.join(DATA, "labels"))
CACHE = os.path.join(DATA, "rail_features.npz")
LABELS = os.path.join(LABELS_DIR, "Rail_Corrugation_Train_Labels.csv")
FAST = 9.7          # m/s; no training file below this is corrugated


def load():
    d = np.load(CACHE, allow_pickle=True)
    X, ids, names = d["X"], d["ids"], list(d["names"])
    lab = {}
    with open(LABELS, newline="") as f:
        for r in csv.DictReader(f):
            lab[r["filename"]] = r["label"]
    tr = np.array([f in lab for f in ids])
    y = np.array([lab[f] for f in ids[tr]])
    return X[tr], y, names, X[~tr], ids[~tr]


def lr(C=3.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced",
                                            max_iter=5000))


def col_groups(names):
    """Index sets for the feature families, so we can ablate them."""
    idx = lambda pred: np.array([i for i, n in enumerate(names) if pred(n)])
    return {
        "speed": idx(lambda n: n == "speed_mps"),
        "contrast": idx(lambda n: "contrast" in n),
        "absolute": idx(lambda n: "contrast" not in n and n != "speed_mps"),
    }


def main():
    X, y, names, _, _ = load()
    g = col_groups(names)
    speed = X[:, g["speed"][0]]
    fast = speed >= FAST

    print("train n=%d  %s" % (len(y), {c: int((y == c).sum()) for c in sorted(set(y))}))
    print("fast subpopulation (speed >= %.1f m/s): n=%d  %s\n"
          % (FAST, fast.sum(), {c: int((y[fast] == c).sum()) for c in sorted(set(y))}))

    everything = np.arange(X.shape[1])
    no_speed = np.array([i for i in everything if i not in set(g["speed"])])
    contrast_only = g["contrast"]

    cands = {
        "current (all 167)": lr(),
        "without speed_mps": lr(),
        "contrast bands only": lr(),
        "speed_mps alone": lr(),
    }
    cols = {
        "current (all 167)": everything,
        "without speed_mps": no_speed,
        "contrast bands only": contrast_only,
        "speed_mps alone": g["speed"],
    }

    print("=== Repeated stratified 5-fold, 10 seeds, macro F1 over ALL files ===")
    allres = {}
    for name, model in cands.items():
        r = V.compare(X[:, cols[name]], y, {name: model}, seeds=range(10))[name]
        allres[name] = r
        print(" ", V.report(name, r))

    print("\n=== Same models, macro F1 scored only on the FAST subpopulation ===")
    print("    (trained on everything outside the fold; this isolates the real task)")
    for name, model in cands.items():
        r = V.compare(X[:, cols[name]], y, {name: model}, seeds=range(10),
                      subset=fast)[name]
        print(" ", V.report(name, r))

    print("\n=== Per-class F1, current model, all files ===")
    for c, (m, s) in V.per_class(X, y, lr()).items():
        print("  %-8s %.4f +/- %.4f" % (c, m, s))

    print("\n=== Nested CV: C chosen inside each outer fold ===")
    grid = {"C=%g" % c: lr(c) for c in (0.1, 0.3, 1, 3, 10, 30)}
    sc, chosen = V.nested_macro_f1(X, y, grid, seeds=range(5))
    print(" ", V.report("nested (all 167 features)", sc))
    print("  configs chosen across outer folds:",
          {k: chosen.count(k) for k in sorted(set(chosen))})


if __name__ == "__main__":
    main()
