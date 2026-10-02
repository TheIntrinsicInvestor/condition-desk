"""Door subsystem: find each open/close cycle, then flag abnormal resistance.

Segmentation is not a modelling problem. The recording contains only the
cycles, not the gaps between them, so consecutive rows are either one sample
apart (20 ms) or separated by a multi-second jump. Splitting on a gap above
50 ms reproduces all 110 official training segments exactly, boundaries and
row counts included.

Because the boundaries come out exact, every correctly labelled cycle scores
an IoU of 1.0 and the predicted segment count equals the true count. The
official IoU-weighted F1 then reduces to plain accuracy over the cycles:

    soft_recall = soft_precision = A / n   ->   harmonic mean = A / n

So the model is tuned for accuracy, not F1.
"""
import csv
from datetime import datetime

import numpy as np

GAP_SECONDS = 0.05          # anything longer than one 20 ms sample is a boundary

NORMAL = "Normal"
ABNORMAL = "Abnormal resistance"


def parse_time(s):
    """'2023-7-5-0-0-3-760' -> datetime. Not zero padded, millisecond last."""
    y, mo, d, h, mi, sec, ms = (int(v) for v in s.split("-"))
    return datetime(y, mo, d, h, mi, sec, ms * 1000)


def load(path):
    """Return (list of row dicts, list of datetimes)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, [parse_time(r["Datetime"]) for r in rows]


def segment(times):
    """Index ranges [start, end) of each cycle, split on timestamp gaps."""
    bounds = [0]
    for i in range(len(times) - 1):
        if (times[i + 1] - times[i]).total_seconds() > GAP_SECONDS:
            bounds.append(i + 1)
    bounds.append(len(times))
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _col(rows, a, b, name):
    out = np.empty(b - a)
    for i in range(a, b):
        try:
            out[i - a] = float(rows[i][name])
        except (TypeError, ValueError):
            out[i - a] = np.nan
    return out


def features(rows, times, a, b):
    """Per-cycle features. Deliberately plain: shape of the motor effort."""
    cur = _col(rows, a, b, "Motor current(mA)")
    vol = _col(rows, a, b, "Motor Voltage(10mV)")
    emf = _col(rows, a, b, "Motor electrodynamic force")
    pos = _col(rows, a, b, "Door leaf position")
    opening = _col(rows, a, b, "Door is opening")
    closing = _col(rows, a, b, "Door is closing")

    dur = (times[b - 1] - times[a]).total_seconds()
    n = b - a
    f = {}
    f["duration"] = dur
    f["n_rows"] = n

    for name, v in (("cur", cur), ("vol", vol), ("emf", emf)):
        f[name + "_mean"] = np.nanmean(v)
        f[name + "_max"] = np.nanmax(v)
        f[name + "_std"] = np.nanstd(v)
        f[name + "_med"] = np.nanmedian(v)
        f[name + "_p90"] = np.nanpercentile(v, 90)
        f[name + "_area"] = np.nansum(v) * 0.02          # integral over time
        f[name + "_rng"] = np.nanmax(v) - np.nanmin(v)

    # where in the cycle the effort peaks, and how sustained it is
    if n > 1 and np.isfinite(cur).any():
        f["cur_argmax_frac"] = float(np.nanargmax(cur)) / n
        thr = np.nanmean(cur) + np.nanstd(cur)
        f["cur_frac_above_1sd"] = float(np.nansum(cur > thr)) / n
        half = n // 2
        f["cur_late_minus_early"] = np.nanmean(cur[half:]) - np.nanmean(cur[:half])
    else:
        f["cur_argmax_frac"] = np.nan
        f["cur_frac_above_1sd"] = np.nan
        f["cur_late_minus_early"] = np.nan

    # travel, and effort per unit of travel
    trav = np.nanmax(pos) - np.nanmin(pos)
    f["pos_range"] = trav
    f["pos_start"] = pos[0] if n else np.nan
    f["pos_end"] = pos[-1] if n else np.nan
    f["cur_per_travel"] = f["cur_area"] / trav if trav > 0 else np.nan
    f["speed"] = trav / dur if dur > 0 else np.nan

    # which operation this was (informational only, never predicted)
    f["is_opening"] = np.nanmean(opening)
    f["is_closing"] = np.nanmean(closing)
    return f


def feature_table(path):
    """(feature matrix, names, segment index ranges, times)."""
    rows, times = load(path)
    segs = segment(times)
    dicts = [features(rows, times, a, b) for a, b in segs]
    names = sorted(dicts[0])
    X = np.array([[d[k] for k in names] for d in dicts], dtype=float)
    return X, names, segs, times


def load_labels(path):
    with open(path, newline="") as f:
        return [(r["start_time"], r["end_time"], r["status"])
                for r in csv.DictReader(f)]


def format_time(t):
    return "%d-%d-%d-%d-%d-%d-%d" % (t.year, t.month, t.day, t.hour,
                                     t.minute, t.second, t.microsecond // 1000)
