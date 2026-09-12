"""Small training-time sklearn models and portable JSON logistic export."""

import json
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from .config import SEED, feature_config
from .portable import linear_score as linear_score  # explicit public re-export


def clip_scaled(x):
    return np.clip(x, -6., 6.)


class SupportSelector(BaseEstimator, TransformerMixin):
    """Discard poorly observed/constant measurements using training inputs only."""

    def __init__(self, min_observed=20):
        self.min_observed = min_observed

    def fit(self, x, y=None):
        a = np.asarray(x, dtype=float)
        self.n_features_in_ = a.shape[1]
        self.support_ = np.array([np.isfinite(col).sum() >= self.min_observed and
                                  len(np.unique(col[np.isfinite(col)])) > 1 for col in a.T])
        if not self.support_.any():
            raise ValueError("No supported nonconstant features in training")
        return self

    def transform(self, x):
        return np.asarray(x, dtype=float)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None else
                           [f"x{i}" for i in range(self.n_features_in_)])
        return names[self.support_]


def make_model(kind="logistic", c=.1, support_min=0):
    if kind == "dummy":
        classifier = DummyClassifier(strategy="prior", random_state=SEED)
    elif kind == "logistic":
        classifier = LogisticRegression(C=c, max_iter=3000, solver="lbfgs", random_state=SEED)
    elif kind == "trees":
        classifier = ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8,
                                          max_features=.75, random_state=SEED, n_jobs=2)
    else:
        raise ValueError("Unknown model")
    steps = [("support", SupportSelector(support_min))] if support_min else []
    return Pipeline(steps + [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(clip_scaled, feature_names_out="one-to-one")),
        ("classifier", classifier),
    ])


def export_linear(pipeline, feature_names, path, metadata, calibrator=None):
    if "support" in pipeline.named_steps:
        feature_names = pipeline.named_steps["support"].get_feature_names_out(feature_names)
    imputer, scaler, clf = (pipeline.named_steps[k] for k in ("imputer", "scaler", "classifier"))
    if not isinstance(clf, LogisticRegression):
        raise ValueError("Only logistic models are exported to the production JSON format")
    data = {"schema_version": 1, "feature_config": feature_config(), "feature_names": list(feature_names),
            "impute": imputer.statistics_.tolist(), "missing_indices": imputer.indicator_.features_.tolist(),
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "clip": 6.,
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "calibration": calibrator or {"slope": 1., "intercept": 0., "method": "none"},
            "threshold": .5, "metadata": metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return data
