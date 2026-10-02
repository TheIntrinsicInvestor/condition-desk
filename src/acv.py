"""ACV subsystem: localise the car with a refrigerant leak, as a ranking.

Physics: a unit low on refrigerant cannot reject heat properly, so it sits
warmer relative to its own cooling setpoint than its seven siblings do. The
ranking feature is therefore each car's mean (indoor temperature - cooling
setpoint), restricted to rows where that car is actually cooling.

Schema notes, all confirmed by reading the files:
  - Column order is scrambled, so everything is looked up by header name.
  - The worksheet is NOT always called "Sheet1" (the test file's sheet is
    named after its original fault case), so always take sheetnames[0].
  - Outdoor temperature appears under two different names across files.
  - acv_case_04 uses a completely different vocabulary (483 columns, 63
    parameters per car) and shares only "ACV Running Mode" with the rest, so
    it needs its own column mapping.
"""
import re

import numpy as np
import openpyxl

# Running modes that mean "this unit is actively cooling".
COOLING_MODES = ("Automatic Cooling", "Full Cooling", "Half Cooling")

# Standard schema (5 of 6 training files, and the held-out test file).
STD = {
    "indoor": "Indoor Average Temperature",
    "setpoint": "ACV Control Temperature (Cooling)",
    "mode": "ACV Running Mode",
    "outdoor": ("Outdoor Average Temperature", "Outside Temperature Sensor Reading"),
}

# acv_case_04's alternative vocabulary.
ALT = {
    "indoor": "Passenger Cabin Temperature Detected Value",
    "setpoint": "Target Temperature Value",
    "mode": "ACV Running Mode",
    "outdoor": ("Fresh Air Temperature Detected Value",),
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(path):
    """Return (header list, rows) from the first worksheet."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]           # never hardcode "Sheet1"
    it = ws.iter_rows(values_only=True)
    hdr = [str(h).strip() if h is not None else "" for h in next(it)]
    rows = list(it)
    wb.close()
    return hdr, rows


def cars_in(hdr):
    return sorted({m.group(1) for h in hdr
                   if (m := re.match(r"^Car (\w+)\s*-", h))})


def pick_schema(hdr):
    """Choose the column vocabulary this file uses."""
    for schema in (STD, ALT):
        if any(schema["indoor"] in h for h in hdr):
            return schema
    raise ValueError("no recognised indoor-temperature column")


def _col(cols, car, name):
    return cols.get("Car %s - %s" % (car, name))


def car_scores(hdr, rows, cooling_only=True, outdoor_weight=False):
    """Mean excess of indoor temperature over the cooling setpoint, per car."""
    schema = pick_schema(hdr)
    cols = {h: i for i, h in enumerate(hdr)}
    out = {}
    for car in cars_in(hdr):
        i_ind = _col(cols, car, schema["indoor"])
        i_set = _col(cols, car, schema["setpoint"])
        if i_ind is None or i_set is None:
            continue
        i_mode = _col(cols, car, schema["mode"])
        i_out = None
        for name in schema["outdoor"]:
            i_out = _col(cols, car, name)
            if i_out is not None:
                break

        ind = np.array([_num(r[i_ind]) for r in rows])
        sp = np.array([_num(r[i_set]) for r in rows])
        keep = ~np.isnan(ind) & ~np.isnan(sp)
        if cooling_only and i_mode is not None:
            modes = np.array([str(r[i_mode]) for r in rows])
            m = np.isin(modes, COOLING_MODES)
            if m.sum() > 50:            # only if the filter leaves real data
                keep &= m
        if not keep.any():
            out[car] = np.nan
            continue

        excess = ind[keep] - sp[keep]
        if outdoor_weight and i_out is not None:
            od = np.array([_num(r[i_out]) for r in rows])[keep]
            w = od - np.nanmin(od)
            if np.nansum(w) > 0:
                out[car] = float(np.nansum(excess * w) / np.nansum(w))
                continue
        out[car] = float(np.nanmean(excess))
    return out


def rank(path, **kw):
    """Cars ordered most to least likely to be the faulty one."""
    hdr, rows = load(path)
    s = car_scores(hdr, rows, **kw)
    ranked = sorted(s, key=lambda c: (-s[c] if not np.isnan(s[c]) else -np.inf))
    return ranked, s
