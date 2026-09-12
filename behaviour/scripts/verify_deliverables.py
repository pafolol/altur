"""Local final inventory/integrity checks; emits aggregate statuses, no call records."""

import hashlib
import json
import re
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
from behavior.config import ROOT, feature_config
from behavior.dataset import load_feature_table, read_manifest
from behavior.metrics import evaluate
from behavior.portable import linear_score
from behavior.vad import MODEL_PATH, MODEL_SHA256


def main():
    required = [
        "behavior/audio.py", "behavior/vad.py", "behavior/events.py", "behavior/features.py",
        "behavior/inference.py", "behavior/service.py", "behavior/README.md", "Dockerfile",
        ".dockerignore", ".env.example", "requirements.lock", "requirements-runtime.lock",
        "artifacts/models/behavior.json", "artifacts/vad/silero_vad.onnx",
        "reports/BEHAVIOUR_REPORT.md", "reports/WHAT_I_NEED_TO_KNOW.md", "reports/experiments.csv",
        "reports/metrics/final_model.json", "reports/metrics/final_family_ablations.json",
        "reports/metrics/prefixes.json", "reports/metrics/robustness.json", "reports/metrics/latency.json",
        "reports/metrics/vad_comparison.json", "reports/metrics/organizer_to_vad_transfer.json",
        "reports/metrics/shortcut_controls.json", "reports/metrics/error_summary.json",
        "reports/metrics/http_smoke.json", "reports/metrics/feature_dictionary.json",
    ]
    missing = [name for name in required if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
    assert not missing, f"Missing deliverables: {missing}"
    artifact = json.loads((ROOT / "artifacts/models/behavior.json").read_text())
    assert artifact["feature_config"] == feature_config()
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256
    assert len(artifact["feature_names"]) == 24 and not any(n.endswith("_count") for n in artifact["feature_names"])
    metadata = read_manifest()
    for source in ("organizer", "vad"):
        assert len(load_feature_table(source)) == len(metadata)
    validation = load_feature_table("vad").query("split == 'val'")
    current_scores = linear_score(artifact, validation[artifact["feature_names"]])
    measured = evaluate(validation.y, current_scores, artifact["threshold"])
    frozen = json.loads((ROOT / "reports/metrics/frozen_evaluation.json").read_text())["val"]
    reported = json.loads((ROOT / "reports/metrics/final_model.json").read_text())["val"]
    for key in ("accuracy", "balanced_accuracy", "roc_auc", "brier", "fpr", "fnr"):
        np.testing.assert_allclose([frozen[key], reported[key]], measured[key], atol=1e-12, rtol=0,
                                   err_msg=f"Stale model/report metric: {key}")
    assert measured["confusion_matrix"] == [[36, 1], [1, 33]]
    # Actual WAV-based full-call predictions must agree with the saved feature/model route.
    prefixes = pd.read_csv(ROOT / "reports/private/prefix_predictions.csv")
    full = prefixes[prefixes.prefix == "full"].set_index("anon_id")
    actual_wav_scores = full.loc[validation.anon_id, "synthetic_probability"].to_numpy()
    np.testing.assert_allclose(actual_wav_scores, current_scores, atol=1e-12, rtol=0)
    feature_dictionary = json.loads((ROOT / "reports/metrics/feature_dictionary.json").read_text())
    assert {d["name"] for d in feature_dictionary} == set(artifact["feature_names"])
    full = (ROOT / "reports/BEHAVIOUR_REPORT.md").read_text()
    assert not re.search(r"\{\{[^}]+\}\}", full)
    assert len(re.findall(r"^## \d+\.", full, re.MULTILINE)) == 24
    assert len((ROOT / "reports/WHAT_I_NEED_TO_KNOW.md").read_text().split()) < len(full.split())
    broken_links = []
    for relative in ("README.md", "behavior/README.md", "reports/BEHAVIOUR_REPORT.md"):
        path = ROOT / relative
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#") and not (path.parent / target.split("#")[0]).exists():
                broken_links.append((relative, target))
    assert not broken_links, f"Broken local documentation links: {broken_links}"
    figures = list((ROOT / "reports/figures").glob("*.png"))
    assert len(figures) >= 14
    for path in figures:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert min(image.size) >= 300
            extrema = image.convert("RGB").getextrema()
            assert any(lo != hi for lo, hi in extrema), "Blank figure"
    # Check actual ignore behavior in the parent home Git repository without modifying it.
    protected = ["manifest.csv", "turns/", "artifacts/", "reports/private/", ".env", "../audio/"]
    for name in protected:
        assert subprocess.run(["git", "check-ignore", "-q", "--", name], cwd=ROOT).returncode == 0, f"Unprotected local data: {name}"
    secret_pattern = re.compile(r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{24,}[\"']")
    source_files = list((ROOT / "behavior").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    assert not any(secret_pattern.search(path.read_text()) for path in source_files)
    result = {"required_files": len(required), "primary_features_documented": len(feature_dictionary),
              "valid_nonblank_figures": len(figures), "official_calls_preserved": len(metadata),
              "documentation_links_valid": True, "confidential_paths_git_ignored": True,
              "vad_checksum_valid": True, "literal_secret_scan_passed": True,
              "saved_model_matches_reported_metrics": True,
              "actual_wav_predictions_match_feature_pipeline": True,
              "container_execution": "not tested: Docker unavailable", "cloud_resources": "none"}
    (ROOT / "reports/metrics/deliverable_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
