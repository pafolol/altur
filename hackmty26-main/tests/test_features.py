import numpy as np
import pytest
from behavior.features import extract_features, summarize, evidence_quality, FEATURE_SETS
from behavior.turns import Turn
from behavior.model import make_model, export_linear, linear_score
from behavior.metrics import evaluate, reliability


def test_no_events_not_fake_latency_zero():
    f, _ = extract_features([Turn(1, 2, 0)], 5)
    assert f["interruption_count"] == 0
    assert np.isnan(f["interruption_stop_latency_mean"])
    assert np.isnan(f["response_latency_std"])
    assert evidence_quality(f)[0] == 0


def test_variability_requires_three_events_and_cv_denominator():
    assert np.isnan(summarize([.2])["std"])
    assert np.isnan(summarize([.2, .3])["std"])
    assert summarize([1, 2, 3])["std"] == 1
    assert summarize([1, 2, 3])["iqr"] == 1
    assert np.isnan(summarize([0, 0, 0])["cv"])
    assert summarize([0, 0, 0])["std"] == 0


def test_complete_features_allowlist_and_time_invariance():
    turns = [Turn(1, 4, 0), Turn(3.5, 5, 1), Turn(5.3, 6, 0)]
    a, _ = extract_features(turns, 7)
    b, _ = extract_features([Turn(t.start + 5, t.end + 5, t.channel) for t in turns], 12)
    for names in FEATURE_SETS.values():
        assert set(names) <= set(a)
        assert not set(names) & {"anon_id", "label", "split", "y", "filename"}
    for name in FEATURE_SETS["consistency"]:
        assert a[name] == pytest.approx(b[name], nan_ok=True)


def test_train_only_imputation_and_portable_model(tmp_path):
    x = np.array([[0., np.nan], [1., 1.], [2., np.nan], [3., 2.], [4., 3.], [5., 4.]])
    model = make_model().fit(x, [0, 0, 0, 1, 1, 1])
    stats = model.named_steps["imputer"].statistics_.copy()
    test = np.array([[1000, np.nan], [np.nan, 3]])
    expected = model.predict_proba(test)[:, 1]
    assert np.array_equal(stats, model.named_steps["imputer"].statistics_)
    data = export_linear(model, ["a", "b"], tmp_path / "model.json", {})
    assert linear_score(data, test) == pytest.approx(expected, abs=1e-12)


def test_metric_orientation_eer_and_probability_bins():
    m = evaluate([0, 0, 1, 1], [.1, .8, .2, .9])
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["fpr"] == m["fnr"] == .5
    assert evaluate([0, 1], [.6, .6])["eer"] == .5
    assert evaluate([0, 1], [.1, .9])["eer"] == 0
    assert reliability([0, 1], [0., 1.])[1] == 0
