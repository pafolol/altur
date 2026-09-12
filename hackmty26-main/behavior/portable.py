"""Minimal NumPy-only runtime for the exported, train-fitted linear pipeline."""

import numpy as np


def linear_score(data, matrix):
    x = np.atleast_2d(np.asarray(matrix, dtype=float))
    missing = np.isnan(x)
    if np.isinf(x).any() or x.shape[1] != len(data["feature_names"]):
        raise ValueError("Incompatible feature vector")
    x = np.where(missing, data["impute"], x)
    x = np.column_stack([x, missing[:, data["missing_indices"]]])
    x = np.clip((x - data["mean"]) / data["scale"], -data["clip"], data["clip"])
    z = x @ np.asarray(data["coef"]) + data["intercept"]
    z = data["calibration"]["slope"] * z + data["calibration"]["intercept"]
    return 1. / (1. + np.exp(-np.clip(z, -40, 40)))
