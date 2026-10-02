"""Rail: does wavelength banding beat frequency banding?

Scores the frequency-banded features (current model), the order-tracked
wavelength features, and their union, on identical folds across 10 seeds, both
over all files and over the fast subpopulation where the task is actually hard.
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
FREQ = os.path.join(DATA, "rail_features.npz")
ORDER = os.path.join(DATA, "rail_order_features.npz")
LABELS = os.path.join(LABELS_DIR, "Rail_Corrugation_Train_Labels.csv")
FAST = 9.7


def lr(C=3.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced",
                                            max_iter=5000))


def load():
    a, b = np.load(FREQ, allow_pickle=True), np.load(ORDER, allow_pickle=True)
    assert list(a["ids"]) == list(b["ids"])
    lab = {}
    with open(LABELS, newline="") as f:
        for r in csv.DictReader(f):
            lab[r["filename"]] = r["label"]
    ids = a["ids"]
    tr = np.array([f in lab for f in ids])
    y = np.array([lab[f] for f in ids[tr]])
    return (a["X"], list(a["names"]), b["X"], list(b["names"]), tr, y, ids)


def main():
    Xf, nf, Xo, no, tr, y, ids = load()
    speed = Xf[tr][:, nf.index("speed_mps")]
    fast = speed >= FAST

    sets = {
        "freq bands (current)": Xf[tr],
        "wavelength bands": Xo[tr],
        "freq + wavelength": np.hstack([Xf[tr], Xo[tr]]),
        "wavelength contrast only": Xo[tr][:, [i for i, n in enumerate(no)
                                               if "wcontrast" in n]],
    }

    print("=== macro F1, stratified 5-fold, 10 seeds, ALL 272 files ===")
    for name, X in sets.items():
        r = V.compare(X, y, {name: lr()}, seeds=range(10))[name]
        print(" ", V.report(name, r))

    print("\n=== same, scored on the FAST subpopulation (n=%d) ===" % fast.sum())
    for name, X in sets.items():
        r = V.compare(X, y, {name: lr()}, seeds=range(10), subset=fast)[name]
        print(" ", V.report(name, r))

    print("\n=== per-class F1, all files ===")
    for name, X in sets.items():
        pc = V.per_class(X, y, lr())
        print("  %-26s " % name +
              "  ".join("%s %.3f" % (c, pc[c][0]) for c in sorted(pc)))


if __name__ == "__main__":
    main()
