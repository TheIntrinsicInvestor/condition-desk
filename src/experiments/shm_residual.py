"""SHM: is the residual tail structure or noise?

54 of the 64 training files land within 5% of their true damage, but the worst
is 13.1%. That is a structured tail rather than a symmetric error, so the
question is whether the residual correlates with anything measurable about the
file. If it does, that is an unmodelled term and the best lead available. If it
does not, the model is at its form-error floor and should be left alone.

Also tests the leads from HANDOFF.md: fitting C per clustered operating
condition (the data spans two lines and two load conditions, AW0 and AW4, but
the files carry no header saying which), and the rainflow residual half-cycle
convention. Every candidate is validated leave-one-out with each choice refit
inside the fold, so the clustering cannot leak.
"""
import csv
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import shm

DATA = os.environ.get("NEBULAX_DATA", "data")
LABELS_DIR = os.environ.get("NEBULAX_LABELS", os.path.join(DATA, "labels"))
TRAIN = os.path.join(DATA, "SHM", "Train")
LABELS = os.path.join(LABELS_DIR, "SHM_Train_Labels.csv")
CACHE = os.path.join(DATA, "shm_cycles.npz")


def load_labels():
    with open(LABELS, newline="") as f:
        return [(r["filename"], float(r["damage"])) for r in csv.DictReader(f)]


def build_cache():
    """Cache per-file cycle sums and gross statistics, so sweeps are instant.

    For a fixed m, damage is (1/C) * sum(n_i * amp_i^m), so caching
    sum(n_i * amp_i^m) per file per m makes every later fit closed-form.
    """
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["names"], d["S"], d["stats"], d["ms"], d["Shalf"]

    labels = load_labels()
    ms = np.arange(3.0, 7.01, 0.5)
    S, stats, names, Shalf = [], [], [], []
    for i, (name, _) in enumerate(labels, 1):
        sig = shm.load_signal(os.path.join(TRAIN, name))
        cyc = list(shm.extract_cycles(shm.reversals(sig)))
        amp = np.array([r / 2.0 for r, _, _ in cyc])
        cnt = np.array([n for _, _, n in cyc])
        mean = np.array([mu for _, mu, _ in cyc])
        ok = amp > 0
        S.append([(cnt[ok] * amp[ok] ** m).sum() for m in ms])
        # variant: count residual half cycles as full cycles instead of 0.5
        cnt_full = np.where(cnt == 0.5, 1.0, cnt)
        Shalf.append([(cnt_full[ok] * amp[ok] ** m).sum() for m in ms])
        stats.append([sig.mean(), sig.std(), len(sig), amp.max(),
                      float(cnt.sum()), float(np.mean(mean)),
                      float(((sig - sig.mean()) ** 3).mean() / (sig.std() ** 3 + 1e-12)),
                      float(amp[ok].mean()), float(np.percentile(amp[ok], 99))])
        names.append(name)
        print("  %2d/%d %s" % (i, len(labels), name), flush=True)

    S, stats, Shalf = np.array(S), np.array(stats), np.array(Shalf)
    np.savez_compressed(CACHE, names=np.array(names), S=S, stats=stats,
                        ms=ms, Shalf=Shalf)
    return np.array(names), S, stats, ms, Shalf


STAT_NAMES = ["mean_stress", "sd_stress", "n_rows", "max_amp", "n_cycles",
              "mean_of_means", "skew", "mean_amp", "p99_amp"]


def loo_mape(s, y):
    """Leave-one-out MAPE for D = s/C, with C refit on the other 63 files.

    C is fitted in log space (the ratio y/s is log-normal-ish), which is what
    the original fit did.
    """
    logr = np.log(s / y)
    errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        C = np.exp(logr[keep].mean())
        errs.append(abs(s[i] / C - y[i]) / y[i])
    return np.array(errs)


def loo_mape_clustered(s, y, feat, k):
    """Same, but C is fitted per cluster, with clustering refit inside the fold."""
    from scipy.cluster.vq import kmeans2
    from sklearn.preprocessing import StandardScaler
    logr = np.log(s / y)
    errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        Z = StandardScaler().fit(feat[keep])
        Ztr = Z.transform(feat[keep])
        cent, lab_tr = kmeans2(Ztr, k, minit="++", seed=0)
        zi = Z.transform(feat[i:i + 1])
        lab_i = int(np.argmin(((cent - zi) ** 2).sum(axis=1)))
        m = lab_tr == lab_i
        C = np.exp(logr[keep][m].mean()) if m.sum() >= 3 else np.exp(logr[keep].mean())
        errs.append(abs(s[i] / C - y[i]) / y[i])
    return np.array(errs)


def main():
    names, S, stats, ms, Shalf = build_cache()
    y = np.array([d for _, d in load_labels()])
    j5 = int(np.argmin(np.abs(ms - 5.0)))
    s5 = S[:, j5]

    base = loo_mape(s5, y)
    print("\n=== Current model (m=5, single C) ===")
    print("  LOO MAPE %.4f%%  -> score %.4f" % (base.mean() * 100, 1 - base.mean()))
    print("  worst %.2f%% on %s" % (base.max() * 100, names[int(np.argmax(base))]))
    print("  within 5%%: %d/64   within 10%%: %d/64"
          % ((base < .05).sum(), (base < .10).sum()))

    print("\n=== Diagnostic: does the LOO residual correlate with file statistics? ===")
    signed = np.log(s5 / np.exp(np.log(s5 / y).mean()) / y)   # signed log residual
    rows = []
    for k, nm in enumerate(STAT_NAMES):
        r = np.corrcoef(stats[:, k], signed)[0, 1]
        ra = np.corrcoef(stats[:, k], base)[0, 1]
        rows.append((abs(r), nm, r, ra))
    for _, nm, r, ra in sorted(rows, reverse=True):
        flag = "  <== structure" if abs(r) > 0.4 else ""
        print("  %-14s corr(signed) %+.3f   corr(abs err) %+.3f%s" % (nm, r, ra, flag))

    print("\n=== Candidate: C fitted per clustered operating condition ===")
    for k in (2, 3, 4):
        e = loo_mape_clustered(s5, y, stats, k)
        print("  k=%d  LOO MAPE %.4f%%  -> score %.4f  (%+.4f vs current)"
              % (k, e.mean() * 100, 1 - e.mean(), base.mean() - e.mean()))

    print("\n=== Candidate: residual half cycles counted as full ===")
    e = loo_mape(Shalf[:, j5], y)
    print("  LOO MAPE %.4f%%  -> score %.4f  (%+.4f vs current)"
          % (e.mean() * 100, 1 - e.mean(), base.mean() - e.mean()))

    print("\n=== Sanity: m re-scanned, C refit per fold ===")
    for j, m in enumerate(ms):
        e = loo_mape(S[:, j], y)
        mark = "  <-- current" if abs(m - 5.0) < 1e-9 else ""
        print("  m=%.1f  LOO MAPE %.4f%%%s" % (m, e.mean() * 100, mark))


if __name__ == "__main__":
    main()
