"""Rail: the fitted classifier, wrapping both feature banks.

Features are the union of two banks computed from the same recording:

  - `rail.features`        frequency band energies (200 Hz bands to Nyquist)
  - `rail_order.features`  wavelength band energies, from order tracking

They are kept as separate banks because they answer different questions. The
frequency bank is what the sensor measures directly. The wavelength bank is
speed-invariant, obtained by resampling each channel at constant distance using
the tacho pulse train, so a corrugation of a given pitch lands in the same band
whatever the train speed. Train speed spans 0 to 19.5 m/s here, so the two are
genuinely different views rather than a reparameterisation.

The union beat frequency alone on 16 of 20 paired cross-validation seeds
(+0.024 macro F1, sign-test p ~ 0.006). Rejected alternatives are recorded in
MODEL_REPORT.md.

Usage, which is what the app should call:

    model, names = fit(train_paths, train_labels)
    label = predict_one(model, path)
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rail
import rail_order

C_REG = 3.0
CLASSES = ["Normal", "Side I", "Side II"]

FIT_NOTES = ("logistic regression, class_weight=balanced, C=3, over the union "
             "of 167 frequency-band and 168 wavelength-band features; "
             "repeated 5-fold macro F1 0.745 +/- 0.027 over 20 seeds")


def features(path):
    """Union of both feature banks for one recording."""
    f1, n1 = rail.features(path)
    f2, n2 = rail_order.features(path)
    return np.concatenate([f1, f2]), list(n1) + list(n2)


def feature_matrix(paths, progress=None):
    rows, names = [], None
    for i, p in enumerate(paths):
        f, names = features(p)
        rows.append(f)
        if progress:
            progress(i + 1, len(paths))
    return np.array(rows), names


def make_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_REG, class_weight="balanced",
                                            max_iter=5000))


def fit(X, y):
    return make_model().fit(X, y)


def predict_one(model, path):
    f, _ = features(path)
    return model.predict(f.reshape(1, -1))[0]


def predict_proba_one(model, path):
    """Class probabilities, for showing confidence in the app."""
    f, _ = features(path)
    p = model.predict_proba(f.reshape(1, -1))[0]
    return dict(zip(model.classes_, p))
