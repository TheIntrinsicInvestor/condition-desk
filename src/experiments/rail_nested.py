"""Rail: honest nested CV over feature set AND regularisation together.

rail_compare.py picked "freq + wavelength" as the best of four options by
looking at the folds it was scored on. That is the exact mistake this project
already made once. Here the feature set and C are both chosen inside each
outer fold, so the returned number was never selected on the data it is
measured on.
"""
import csv
import os
import sys

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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


class ColumnSelect(BaseEstimator, TransformerMixin):
    """Pick a fixed column slice, so feature-set choice can live in the grid."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if self.cols is None else X[:, self.cols]


def load():
    a, b = np.load(FREQ, allow_pickle=True), np.load(ORDER, allow_pickle=True)
    assert list(a["ids"]) == list(b["ids"])
    lab = {}
    with open(LABELS, newline="") as f:
        for r in csv.DictReader(f):
            lab[r["filename"]] = r["label"]
    ids = a["ids"]
    tr = np.array([f in lab for f in ids])
    X = np.hstack([a["X"], b["X"]])
    names = list(a["names"]) + list(b["names"])
    y = np.array([lab[f] for f in ids[tr]])
    return X, names, tr, y, ids


def main():
    X, names, tr, y, ids = load()
    n_freq = 167
    col = {
        "freq": np.arange(n_freq),
        "wave": np.arange(n_freq, X.shape[1]),
        "both": np.arange(X.shape[1]),
    }

    grid = {}
    for fname, cols in col.items():
        for C in (0.1, 0.3, 1, 3, 10):
            grid["%s/C=%g" % (fname, C)] = make_pipeline(
                ColumnSelect(cols), StandardScaler(),
                LogisticRegression(C=C, class_weight="balanced", max_iter=5000))

    print("Nested CV: feature set and C both selected inside each outer fold")
    print("grid size: %d configs\n" % len(grid))
    sc, chosen = V.nested_macro_f1(X[tr], y, grid, seeds=range(5))
    print(" ", V.report("NESTED (honest)", sc))

    counts = {}
    for c in chosen:
        counts[c] = counts.get(c, 0) + 1
    print("\n  configs chosen across the 25 outer folds:")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print("    %-12s %d" % (k, counts[k]))
    fam = {}
    for c in chosen:
        f = c.split("/")[0]
        fam[f] = fam.get(f, 0) + 1
    print("  by feature set:", fam)


if __name__ == "__main__":
    main()
