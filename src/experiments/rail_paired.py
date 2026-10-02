"""Rail: paired per-seed comparison, plus the two-binary decomposition.

A difference in means across seeds is weak evidence when the spread is this
wide. A paired comparison on identical folds is much stronger: if the union
wins on 19 of 20 seeds, that is hard to explain as noise even though the means
overlap by one standard deviation.

Also tests the decomposition the info kit implies. The two rails are judged
independently, so "is Side I corrugated?" and "is Side II corrugated?" is the
true shape of the question. Each binary model then learns from all 272 files
instead of splitting the evidence three ways, which should help Side I most,
and Side I is a third of macro F1 estimated from only 14 examples.
"""
import csv
import os
import sys

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate as V

DATA = os.environ.get("NEBULAX_DATA", "data")
LABELS_DIR = os.environ.get("NEBULAX_LABELS", os.path.join(DATA, "labels"))
FREQ = os.path.join(DATA, "rail_features.npz")
ORDER = os.path.join(DATA, "rail_order_features.npz")
LABELS = os.path.join(LABELS_DIR, "Rail_Corrugation_Train_Labels.csv")
SEEDS = range(20)


def lr(C=3.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced",
                                            max_iter=5000))


def load():
    a, b = np.load(FREQ, allow_pickle=True), np.load(ORDER, allow_pickle=True)
    lab = {}
    with open(LABELS, newline="") as f:
        for r in csv.DictReader(f):
            lab[r["filename"]] = r["label"]
    ids = a["ids"]
    tr = np.array([f in lab for f in ids])
    y = np.array([lab[f] for f in ids[tr]])
    return a["X"], b["X"], tr, y


def binary_predict(Xf, y, fold_ids, C=3.0):
    """Two independent binary models, recombined into the 3-class label.

    Side I and Side II are predicted separately from all 272 files. If both
    fire, the more confident one wins; if neither fires, the file is Normal.
    """
    pred = np.empty(len(y), dtype=object)
    for k in np.unique(fold_ids):
        te = fold_ids == k
        probs = {}
        for side in ("Side I", "Side II"):
            yb = (y == side).astype(int)
            m = clone(lr(C)).fit(Xf[~te], yb[~te])
            probs[side] = m.predict_proba(Xf[te])[:, 1]
        out = []
        for i in range(int(te.sum())):
            p1, p2 = probs["Side I"][i], probs["Side II"][i]
            if p1 < 0.5 and p2 < 0.5:
                out.append("Normal")
            else:
                out.append("Side I" if p1 >= p2 else "Side II")
        pred[te] = out
    return pred


def main():
    Xf, Xo, tr, y = load()
    Xf, Xo = Xf[tr], Xo[tr]
    Xb = np.hstack([Xf, Xo])

    print("=== Paired per-seed comparison, identical folds, %d seeds ===" % len(SEEDS))
    res = {}
    for name, X in (("freq", Xf), ("wave", Xo), ("both", Xb)):
        res[name] = V.compare(X, y, {name: lr()}, seeds=SEEDS)[name]
        print(" ", V.report(name, res[name]))

    for a, b in (("both", "freq"), ("wave", "freq")):
        d = res[a] - res[b]
        wins = int((d > 0).sum())
        print("\n  %s minus %s: mean %+.4f, sd %.4f, wins %d/%d seeds"
              % (a, b, d.mean(), d.std(), wins, len(d)))
        print("    per-seed diffs:", " ".join("%+.3f" % v for v in d))

    print("\n=== Two-binary decomposition vs 3-class, on identical folds ===")
    for name, X in (("freq", Xf), ("both", Xb)):
        three, two = [], []
        for seed in SEEDS:
            fid = V.folds(y, seed, 5)
            three.append(f1_score(y, V.cv_predict(X, y, lr(), fid), average="macro"))
            two.append(f1_score(y, binary_predict(X, y, fid), average="macro"))
        three, two = np.array(three), np.array(two)
        d = two - three
        print(" ", V.report("%s 3-class" % name, three))
        print(" ", V.report("%s 2-binary" % name, two))
        print("    2-binary minus 3-class: %+.4f, wins %d/%d"
              % (d.mean(), int((d > 0).sum()), len(d)))

    print("\n=== Side I F1 specifically (the bottleneck) ===")
    for name, X in (("freq", Xf), ("wave", Xo), ("both", Xb)):
        pc = V.per_class(X, y, lr(), seeds=SEEDS)
        print("  %-6s Side I %.4f +/- %.4f   Side II %.4f   Normal %.4f"
              % (name, pc["Side I"][0], pc["Side I"][1],
                 pc["Side II"][0], pc["Normal"][0]))


if __name__ == "__main__":
    main()
