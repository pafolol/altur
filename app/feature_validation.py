import math

import numpy as np


def validate_finite_features(features):
    """Reject non-finite values before any model sees request-derived data."""
    invalid = [name for name, value in features.items() if not math.isfinite(float(value))]
    if invalid:
        raise ValueError(f"Features no finitas: {', '.join(invalid)}")
    return features
