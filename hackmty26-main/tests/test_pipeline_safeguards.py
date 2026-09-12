import json
import numpy as np
import pytest
from behavior.config import ROOT
from behavior.dataset import load_feature_table
from behavior.features import extract_features
from behavior.model import SupportSelector
from behavior.inference import BehaviorDetector
from behavior.turns import Turn


def test_support_selection_fits_only_available_training_data():
    x = np.array([[1, np.nan, 5], [2, np.nan, 5], [3, 4, 5]], dtype=float)
    selector = SupportSelector(min_observed=2).fit(x)
    assert selector.support_.tolist() == [True, False, False]
    assert selector.transform([[1000, 2000, 3000]]).tolist() == [[1000]]


def test_append_silence_does_not_change_final_model_inputs():
    artifact_path = ROOT / "artifacts/models/behavior.json"
    if not artifact_path.exists():
        pytest.skip("Final artifact required")
    names = json.loads(artifact_path.read_text())["feature_names"]
    assert not any(n.endswith("_count") or n.startswith("agent_turn_") for n in names)
    assert not set(names) & {"duration_s", "leading_silence_s", "trailing_silence_s"}
    turns = [Turn(1, 4, 0), Turn(3.5, 5, 1), Turn(5.3, 6, 0)]
    a, _ = extract_features(turns, 7)
    b, _ = extract_features(turns, 40)
    for name in names:
        assert a[name] == pytest.approx(b[name], nan_ok=True)


@pytest.mark.integration
def test_feature_cache_checks_official_split():
    if not (ROOT / "artifacts/features/vad.meta.json").exists():
        pytest.skip("Feature cache required")
    df = load_feature_table("vad")
    assert len(df) == 353
    assert (df.split == "train").sum() == 282 and (df.split == "val").sum() == 71


@pytest.mark.integration
def test_feature_version_mismatch_rejected_before_vad_load(tmp_path):
    source = ROOT / "artifacts/models/behavior.json"
    if not source.exists():
        pytest.skip("Final artifact required")
    data = json.loads(source.read_text())
    data["feature_config"]["version"] = "invalid"
    target = tmp_path / "bad.json"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="configuration mismatch"):
        BehaviorDetector(target)
