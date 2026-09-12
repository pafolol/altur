"""Train-only out-of-fold sigmoid calibration; no validation fitting."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from .config import SEED


def fit_oof_calibrator(pipeline, x_train, y_train):
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    # Entire pipeline cloned: each fold fits its OWN imputer and scaler.
    logits = cross_val_predict(clone(pipeline), x_train, y_train, cv=folds,
                               method="decision_function", n_jobs=1)
    sigmoid = LogisticRegression(C=1., solver="lbfgs", random_state=SEED)
    sigmoid.fit(logits.reshape(-1, 1), y_train)
    slope = float(sigmoid.coef_[0, 0])
    if slope <= 0:
        raise ValueError("Non-monotone calibrator: check whether training signal exists")
    return {"method": "train_5fold_oof_sigmoid", "slope": slope,
            "intercept": float(sigmoid.intercept_[0]), "fold_seed": SEED,
            "warning": "No speaker groups available; internal folds may share callers"}, logits


def apply_calibrator(calibrator, logits):
    z = calibrator["slope"] * np.asarray(logits) + calibrator["intercept"]
    return 1. / (1. + np.exp(-np.clip(z, -40, 40)))
