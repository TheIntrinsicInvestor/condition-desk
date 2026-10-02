"""Shared validation protocol for every model experiment.

Exists because the project's one standing rule is "refit every choice inside
the validation fold", and the rail sweep already broke it once, reading 0.739
against a true 0.721. Centralising the protocol makes breaking it harder than
following it.

Two guarantees this module provides:

  1. Every candidate in a comparison is scored on *identical* folds, so the
     difference between two candidates is paired and its spread is meaningful.
  2. `nested_macro_f1` selects the config inside each outer fold, so the number
     it returns was never chosen on the data it is measured on.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def folds(y, seed, n_splits=5):
    """Fold assignment as an array of fold indices, stratified on y."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.empty(len(y), dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        out[te] = k
    return out


def cv_predict(X, y, model, fold_ids):
    """Out-of-fold predictions using one fixed fold assignment."""
    pred = np.empty(len(y), dtype=object)
    for k in np.unique(fold_ids):
        te = fold_ids == k
        m = clone(model).fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    return pred


def compare(X, y, candidates, seeds=range(10), n_splits=5, subset=None):
    """Score every candidate on identical folds, across seeds.

    candidates: {name: estimator}. subset: optional boolean mask selecting the
    rows the score is computed over (models still train on everything outside
    the fold, which is the honest thing to do when the subset is a population
    of interest rather than a different task).

    Returns {name: array of per-seed macro F1}.
    """
    out = {name: [] for name in candidates}
    for seed in seeds:
        fold_ids = folds(y, seed, n_splits)
        for name, model in candidates.items():
            pred = cv_predict(X, y, model, fold_ids)
            m = np.ones(len(y), dtype=bool) if subset is None else subset
            out[name].append(f1_score(y[m], pred[m], average="macro"))
    return {k: np.array(v) for k, v in out.items()}


def nested_macro_f1(X, y, grid, seeds=range(5), outer=5, inner=4):
    """Unbiased macro F1: the config is selected inside each outer fold.

    grid: {name: estimator}. Returns (per-seed scores, list of chosen names).
    """
    scores, chosen = [], []
    for seed in seeds:
        fold_ids = folds(y, seed, outer)
        pred = np.empty(len(y), dtype=object)
        for k in np.unique(fold_ids):
            te = fold_ids == k
            Xtr, ytr = X[~te], y[~te]
            # select on the training part only
            inner_ids = folds(ytr, seed + 1000, inner)
            best, best_s = None, -np.inf
            for name, model in grid.items():
                p = cv_predict(Xtr, ytr, model, inner_ids)
                s = f1_score(ytr, p, average="macro")
                if s > best_s:
                    best, best_s = name, s
            chosen.append(best)
            pred[te] = clone(grid[best]).fit(Xtr, ytr).predict(X[te])
        scores.append(f1_score(y, pred, average="macro"))
    return np.array(scores), chosen


def report(name, scores):
    return "%-34s %.4f +/- %.4f  (min %.4f max %.4f)" % (
        name, scores.mean(), scores.std(), scores.min(), scores.max())


def per_class(X, y, model, seeds=range(10), n_splits=5):
    """Per-class F1 averaged across seeds, to see where the macro score lives."""
    classes = sorted(set(y))
    acc = {c: [] for c in classes}
    for seed in seeds:
        pred = cv_predict(X, y, model, folds(y, seed, n_splits))
        f = f1_score(y, pred, average=None, labels=classes)
        for c, v in zip(classes, f):
            acc[c].append(v)
    return {c: (np.mean(v), np.std(v)) for c, v in acc.items()}
