"""Door: stream-relative features, and what cycle 32 actually is.

The training set is perfectly separable on vol_area, so every reasonable model
scores 1.000 in cross-validation. That is not a passing grade, it is a blind
test: CV here cannot rank models, tune anything, or detect a problem.

So this module does not try to win a CV comparison, because none is available.
It asks two questions that CV cannot:

  1. Is there an absolute offset between the training stream and the test
     stream? If the test door, supply voltage or sensor gain differs at all,
     an absolute threshold fitted on the training stream lands in the wrong
     place, and nothing in cross-validation would reveal it. Dividing each
     cycle's features by the median across its own stream removes any such
     offset. This uses no labels and the whole test stream is available at
     prediction time, so it is legitimate.

  2. What is cycle 32? It is the one test cycle that does not fit the clean
     two-cluster story. Summary statistics have been argued to a draw, so the
     honest move is to look at where it sits in both streams.

Per the accept rule, a change here ships only with a stated mechanism, an
unchanged training score, and few enough flipped test predictions that each
can be inspected by hand.
"""
import csv
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import door

DATA = os.environ.get("NEBULAX_DATA", "data")
LABELS_DIR = os.environ.get("NEBULAX_LABELS", os.path.join(DATA, "labels"))
TRAIN = os.path.join(DATA, "Door", "Train.csv")
TEST = os.path.join(DATA, "Door", "Test.csv")
LABELS = os.path.join(LABELS_DIR, "Door_Train_Segments_Answer.csv")
KEY = "vol_area"


def load_train():
    X, names, segs, times = door.feature_table(TRAIN)
    lab = door.load_labels(LABELS)
    status = [s for _, _, s in lab]
    assert len(status) == len(segs), "%d labels vs %d segments" % (len(status), len(segs))
    return X, names, np.array(status), segs, times


def relative(X, names):
    """Divide each column by the median over the stream. Sign-safe columns only.

    Columns that can be zero or negative (differences, fractions, positions)
    are left alone, because dividing them by a near-zero median manufactures
    outliers rather than removing an offset.
    """
    Xr = X.copy()
    scaled = []
    for j, n in enumerate(names):
        col = X[:, j]
        med = np.nanmedian(col)
        if np.isfinite(med) and med > 1e-9 and np.nanmin(col) >= 0:
            Xr[:, j] = col / med
            scaled.append(n)
    return Xr, scaled


def main():
    Xtr, names, y, segs_tr, _ = load_train()
    Xte, names2, segs_te, times_te = door.feature_table(TEST)
    assert names == names2
    k = names.index(KEY)

    print("=== The separating feature, raw ===")
    a = Xtr[y == door.NORMAL, k]
    b = Xtr[y == door.ABNORMAL, k]
    print("  train Normal   n=%3d  %.0f .. %.0f" % (len(a), a.min(), a.max()))
    print("  train Abnormal n=%3d  %.0f .. %.0f" % (len(b), b.min(), b.max()))
    print("  gap: %.0f .. %.0f  -> threshold %.0f"
          % (a.max(), b.min(), (a.max() + b.min()) / 2))
    print("  test           n=%3d  %.0f .. %.0f"
          % (len(Xte), Xte[:, k].min(), Xte[:, k].max()))

    print("\n=== Stream medians: is there an offset between the streams? ===")
    for nm in ("vol_area", "cur_area", "vol_mean", "cur_mean", "emf_mean",
               "duration", "pos_range"):
        j = names.index(nm)
        mt, me = np.nanmedian(Xtr[:, j]), np.nanmedian(Xte[:, j])
        print("  %-10s train %12.2f   test %12.2f   ratio %.4f"
              % (nm, mt, me, me / mt if mt else np.nan))

    thr = (a.max() + b.min()) / 2
    pred_abs = np.where(Xte[:, k] > thr, door.ABNORMAL, door.NORMAL)

    Xtr_r, scaled = relative(Xtr, names)
    Xte_r, _ = relative(Xte, names)
    ar, br = Xtr_r[y == door.NORMAL, k], Xtr_r[y == door.ABNORMAL, k]
    print("\n=== Stream-relative version of the same feature ===")
    print("  %d of %d columns rescaled" % (len(scaled), len(names)))
    print("  train Normal   %.4f .. %.4f" % (ar.min(), ar.max()))
    print("  train Abnormal %.4f .. %.4f" % (br.min(), br.max()))
    sep = ar.max() < br.min()
    print("  still perfectly separable: %s" % sep)
    thr_r = (ar.max() + br.min()) / 2
    pred_rel = np.where(Xte_r[:, k] > thr_r, door.ABNORMAL, door.NORMAL)

    print("\n=== Do the two disagree on any test cycle? ===")
    diff = np.flatnonzero(pred_abs != pred_rel)
    print("  absolute threshold : %d Abnormal of %d"
          % ((pred_abs == door.ABNORMAL).sum(), len(pred_abs)))
    print("  stream-relative    : %d Abnormal of %d"
          % ((pred_rel == door.ABNORMAL).sum(), len(pred_rel)))
    print("  disagreements: %d %s" % (len(diff), list(diff + 1)))

    print("\n=== Every test cycle, sorted by the separating feature ===")
    print("  margin is distance from the threshold, in units of the training gap")
    gap = b.min() - a.max()
    order = np.argsort(Xte[:, k])
    for i in order:
        margin = (Xte[i, k] - thr) / gap
        flag = ""
        if abs(margin) < 0.5:
            flag = "   <== borderline"
        if i == 31:
            flag += "   <== cycle 32"
        print("  cycle %2d  %s=%8.0f  margin %+6.2f  %-20s%s"
              % (i + 1, KEY, Xte[i, k], margin, pred_abs[i], flag))


if __name__ == "__main__":
    main()
