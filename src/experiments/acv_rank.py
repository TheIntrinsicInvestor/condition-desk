"""ACV: rank aggregation across several physically-motivated features.

The current model bets everything on one feature, mean(indoor - cooling
setpoint), chosen because it widened margins on the same five cases it is
validated against. With five observations that is selection on the validation
set, and two of the five were near misses (case_05 separated 0.713 from 0.731,
a margin of 0.018).

Rank aggregation is the standard remedy: score each car by several features
that are independently motivated by the physics, convert each to a rank, and
average. It cannot be defeated by one feature being lucky, which is the live
risk here. It also cannot be tuned, which is the point: with five cases there
is nothing honest to tune on.

The features, and why each should be high for a car losing refrigerant:

  excess       mean(indoor - cooling setpoint) while cooling. The direct
               statement that the car is not holding its setpoint.
  pulldown     how slowly indoor temperature falls while cooling. A unit low
               on refrigerant has reduced cooling capacity, so it pulls the
               cabin down more slowly for the same demand.
  duty         fraction of time spent in the harder cooling modes. A starved
               unit runs harder to achieve the same result.
  vs_fleet     mean indoor temperature minus the median across the car's
               siblings in the same file. Robust to a file-wide offset in
               ambient conditions, which absolute thresholds are not.
  ambient      correlation of indoor with outdoor temperature. A unit that
               cannot reject heat tracks ambient instead of its setpoint.

Scoring is (n - (r - 1)) / n, so slipping from rank 1 to rank 2 costs only
0.125 while a catastrophic miss to rank 7 costs 0.75. That asymmetry is why
trading a little top-1 sharpness for robustness is the right bet.
"""
import csv
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import acv

DATA = os.environ.get("NEBULAX_DATA", "data")
LABELS_DIR = os.environ.get("NEBULAX_LABELS", os.path.join(DATA, "labels"))
TRAIN = os.path.join(DATA, "ACV", "Train")
TEST = os.path.join(DATA, "ACV", "Test")
LABELS = os.path.join(LABELS_DIR, "ACV_Train_Labels.csv")

# Modes ordered by how hard the unit is working.
HARD_MODES = ("Full Cooling",)
ANY_COOLING = acv.COOLING_MODES

FEATURES = ["excess", "pulldown", "duty", "vs_fleet", "ambient"]


def car_series(hdr, rows):
    """Per-car aligned arrays: indoor, setpoint, mode, outdoor."""
    schema = acv.pick_schema(hdr)
    cols = {h: i for i, h in enumerate(hdr)}
    out = {}
    for car in acv.cars_in(hdr):
        i_ind = cols.get("Car %s - %s" % (car, schema["indoor"]))
        i_set = cols.get("Car %s - %s" % (car, schema["setpoint"]))
        if i_ind is None or i_set is None:
            continue
        i_mode = cols.get("Car %s - %s" % (car, schema["mode"]))
        i_out = None
        for nm in schema["outdoor"]:
            if ("Car %s - %s" % (car, nm)) in cols:
                i_out = cols["Car %s - %s" % (car, nm)]
                break
        g = lambda i: (np.array([acv._num(r[i]) for r in rows])
                       if i is not None else None)
        mode = (np.array([str(r[i_mode]) for r in rows])
                if i_mode is not None else None)
        out[car] = dict(indoor=g(i_ind), setpoint=g(i_set), outdoor=g(i_out),
                        mode=mode)
    return out


def _cooling_mask(d):
    if d["mode"] is None:
        return np.ones(len(d["indoor"]), dtype=bool)
    m = np.isin(d["mode"], ANY_COOLING)
    return m if m.sum() > 50 else np.ones(len(d["indoor"]), dtype=bool)


def compute(series):
    """{feature: {car: value}}, higher meaning more likely faulty."""
    cars = list(series)
    n = len(series[cars[0]]["indoor"])
    fleet_median = np.nanmedian(
        np.vstack([series[c]["indoor"] for c in cars]), axis=0)

    f = {k: {} for k in FEATURES}
    for c in cars:
        d = series[c]
        ind, sp = d["indoor"], d["setpoint"]
        cool = _cooling_mask(d) & ~np.isnan(ind) & ~np.isnan(sp)

        f["excess"][c] = (float(np.nanmean(ind[cool] - sp[cool]))
                          if cool.any() else np.nan)

        # pulldown: mean rate of change of indoor temperature while cooling,
        # sign flipped so that "falls slowly" scores high
        if cool.sum() > 10:
            di = np.diff(ind)
            w = cool[1:] & np.isfinite(di)
            f["pulldown"][c] = float(np.nanmean(di[w])) if w.any() else np.nan
        else:
            f["pulldown"][c] = np.nan

        if d["mode"] is not None:
            f["duty"][c] = float(np.isin(d["mode"], HARD_MODES).mean())
        else:
            f["duty"][c] = np.nan

        ok = np.isfinite(ind) & np.isfinite(fleet_median)
        f["vs_fleet"][c] = (float(np.nanmean(ind[ok] - fleet_median[ok]))
                            if ok.any() else np.nan)

        od = d["outdoor"]
        if od is not None:
            ok = np.isfinite(ind) & np.isfinite(od)
            if ok.sum() > 10 and np.std(od[ok]) > 1e-6 and np.std(ind[ok]) > 1e-6:
                f["ambient"][c] = float(np.corrcoef(ind[ok], od[ok])[0, 1])
            else:
                f["ambient"][c] = np.nan
        else:
            f["ambient"][c] = np.nan
    return f


def to_ranks(scores):
    """Rank cars 1..n by a score, descending. NaN goes last, ties share."""
    cars = list(scores)
    vals = np.array([scores[c] if np.isfinite(scores[c]) else -np.inf
                     for c in cars])
    order = np.argsort(-vals, kind="stable")
    r = np.empty(len(cars))
    r[order] = np.arange(1, len(cars) + 1)
    return {c: float(r[i]) for i, c in enumerate(cars)}


def aggregate(path, use=FEATURES):
    """(ranked car list, per-feature ranks, mean rank per car)."""
    hdr, rows = acv.load(path)
    f = compute(car_series(hdr, rows))
    usable = [k for k in use if any(np.isfinite(v) for v in f[k].values())]
    per = {k: to_ranks(f[k]) for k in usable}
    cars = list(f["excess"])
    mean_rank = {c: float(np.mean([per[k][c] for k in usable])) for c in cars}
    ranked = sorted(cars, key=lambda c: (mean_rank[c], c))
    return ranked, per, mean_rank, usable


def score(rank, n):
    return (n - (rank - 1)) / n


def main():
    with open(LABELS, newline="") as fh:
        labels = [(r["filename"], r["faulty_car"]) for r in csv.DictReader(fh)]

    print("=== Per-case rank of the TRUE faulty car, by each feature alone ===")
    hdr_row = "%-18s %-6s " % ("case", "true") + \
              " ".join("%-9s" % k for k in FEATURES) + " | AGG  score"
    print(hdr_row)
    print("-" * len(hdr_row))

    agg_scores, single_scores = [], {k: [] for k in FEATURES}
    for name, true in labels:
        path = os.path.join(TRAIN, name)
        try:
            ranked, per, mean_rank, usable = aggregate(path)
        except Exception as e:
            print("%-18s SKIPPED (%s)" % (name, e))
            continue
        n = len(ranked)
        cells = []
        for k in FEATURES:
            if k in per and true in per[k]:
                r = per[k][true]
                cells.append("%d/%d" % (r, n))
                single_scores[k].append(score(r, n))
            else:
                cells.append("-")
        r_agg = ranked.index(true) + 1 if true in ranked else n + 1
        s = score(r_agg, n) if true in ranked else 0.0
        agg_scores.append(s)
        print("%-18s %-6s " % (name, true) +
              " ".join("%-9s" % c for c in cells) +
              " | %d/%d %.3f" % (r_agg, n, s))

    print("\n=== Mean score per method (higher is better, max 1.0) ===")
    for k in FEATURES:
        v = single_scores[k]
        if v:
            print("  %-10s %.4f   over %d cases   (top-1 on %d)"
                  % (k, np.mean(v), len(v), sum(1 for x in v if x == 1.0)))
    print("  %-10s %.4f   over %d cases   (top-1 on %d)"
          % ("AGGREGATE", np.mean(agg_scores), len(agg_scores),
             sum(1 for x in agg_scores if x == 1.0)))

    print("\n=== Test file: do the features agree? ===")
    tpath = os.path.join(TEST, "acv_test_case.xlsx")
    ranked, per, mean_rank, usable = aggregate(tpath)
    print("  usable features:", usable)
    cars = sorted(mean_rank)
    print("  %-6s " % "car" + " ".join("%-9s" % k for k in usable) + " meanrank")
    for c in cars:
        print("  %-6s " % c +
              " ".join("%-9d" % per[k][c] for k in usable) +
              " %.2f" % mean_rank[c])
    print("\n  aggregate ranking: " + "|".join(ranked))
    firsts = [min(per[k], key=lambda c: per[k][c]) for k in usable]
    print("  each feature's top pick:", dict(zip(usable, firsts)))


if __name__ == "__main__":
    main()
