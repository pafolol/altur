"""Issue the integration freeze from completed delta evidence. Does not train/evaluate models."""

from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd

from behavior.config import ROOT
from behavior.decision import DECISION_POLICY_VERSION
from scripts.common import write_json

DELTA = ROOT / "reports/metrics/delta_review"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    manifest_path = ROOT / "reports/FINAL_BEHAVIOR_MANIFEST.json"
    if manifest_path.exists():
        raise SystemExit("Final delta freeze already issued; stop rather than start another cycle.")
    checks = load(DELTA / "checks.json")
    artifact_path = ROOT / checks["artifact_path"]
    assert sha(artifact_path) == checks["artifact_sha256"]
    artifact = load(artifact_path)
    baseline_path = ROOT / checks["reference_baseline"] / "REFERENCE_BASELINE.json"
    baseline = load(baseline_path)
    assert checks["artifact_sha256"] == baseline["deployed_artifact_sha256"]

    # Parse the single final full-suite run, not the earlier targeted regressions.
    xml_path = DELTA / "final_pytest.xml"
    tree = ET.parse(xml_path).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
    totals = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    assert totals["tests"] >= 41 and totals["failures"] == totals["errors"] == totals["skipped"] == 0
    tests = {**totals, "passed": totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"],
             "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q --junitxml=reports/metrics/delta_review/final_pytest.xml",
             "junit_path": str(xml_path.relative_to(ROOT)), "junit_sha256": sha(xml_path),
             "complete_suite_runs_in_delta_review": 1,
             "earlier_targeted_regression": "12 targeted tests; initially 11 pass /1 malformed-RIFF failure, then all 12 pass after minimal parser fix"}
    write_json(DELTA / "final_tests.json", tests)
    isolated = load(DELTA / "isolated_runtime.json")
    http = load(DELTA / "http_smoke.json")
    assert isolated["artifact_sha256"] == checks["artifact_sha256"]
    assert isolated["network_egress_denied"] and isolated["training_data_access_denied"]
    assert isolated["no_sklearn_pandas_torch"] and isolated["valid_wav_inference"] == "passed"
    assert http["health"] == "ready" and http["silence_nonflag_fallback_and_evidence_header"]
    assert http["invalid_input_rejected_without_echo"]
    assert set(http["real_wav_endpoints_passed"]) == {"behavior", "detect"}

    # Log exactly the two predeclared nuisance diagnostics; never update a candidate choice.
    diagnostics = load(DELTA / "nuisance_diagnostics.json")
    log_path = ROOT / "reports/experiments.csv"
    log = pd.read_csv(log_path)
    new_rows = []
    for name, result in diagnostics.items():
        exp = "DELTA_" + name
        assert exp not in set(log.experiment)
        new_rows.append({"source": "delta_diagnostic", "experiment": exp,
                         "hypothesis": "Fixed ASTRA2 nuisance diagnostic; never a deployment candidate",
                         "feature_set": name, "feature_count": len(result["input_features"]),
                         "model": "logistic", "c": .1,
                         **{f"{split}_{key}": result[split][key] for split in ("train", "val")
                            for key in ("accuracy", "balanced_accuracy", "roc_auc", "brier")},
                         "training_s": result["runtime_s"],
                         "conclusion": "Report predictive nuisance information; no deployed artifact/threshold changes",
                         "decision": "DOCUMENT LIMITATION; diagnostic only; frozen artifact unchanged"})
    pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True).to_csv(log_path, index=False)

    deployment_files = sorted(list((ROOT / "behavior").glob("*.py")) +
                              [ROOT / p for p in ("Dockerfile", ".dockerignore", "requirements-runtime.lock")])
    runtime_hashes = {str(p.relative_to(ROOT)): sha(p) for p in deployment_files}
    runtime_revision = hashlib.sha256(json.dumps(runtime_hashes, sort_keys=True).encode()).hexdigest()
    source_files = sorted(deployment_files + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py")))
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    source_revision = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    changed_runtime = [p for p, digest in runtime_hashes.items() if baseline["files"].get(p, {}).get("sha256") != digest]
    metrics_files = [ROOT / "reports/metrics" / p for p in
                     ("final_model.json", "frozen_evaluation.json", "prefixes.json", "latency.json",
                      "final_family_ablations.json", "organizer_to_vad_transfer.json", "robustness.json")]
    assert all(sha(p) == baseline["files"][str(p.relative_to(ROOT))]["sha256"] for p in metrics_files)
    assert sha(ROOT / "artifacts/vad/silero_vad.onnx") == baseline["files"]["artifacts/vad/silero_vad.onnx"]["sha256"]
    latency = load(ROOT / "reports/metrics/latency.json")
    now = datetime.now(timezone.utc).isoformat()
    required_versions = [line.split("==")[0] for line in (ROOT / "requirements-runtime.lock").read_text().splitlines()
                         if re.match(r"^[a-zA-Z0-9-]+==", line)]
    version_command = "import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in " + repr(required_versions) + "}))"
    runtime_versions = json.loads(subprocess.check_output([str(ROOT / ".venv-runtime/bin/python"), "-I", "-c", version_command], text=True))
    record = {
        "freeze_timestamp_utc": now, "ready_for_integration": True,
        "readiness_meaning": "engineering-ready for team integration under the documented development-validation evidence",
        "not_claimed": "proven robustness on Altur hidden set or production banking traffic; verified Docker/cloud deployment",
        "artifact_path": checks["artifact_path"], "artifact_sha256": checks["artifact_sha256"],
        "artifact_changed_during_delta_review": False, "artifact_created_utc": artifact["metadata"]["date_utc"],
        "baseline_record": str(baseline_path.relative_to(ROOT)), "baseline_record_sha256": sha(baseline_path),
        "audit_path": checks["audit_path"], "audit_sha256": checks["audit_sha256"],
        "feature_count": len(artifact["feature_names"]), "feature_names": artifact["feature_names"],
        "missingness_indicators": [artifact["feature_names"][i] for i in artifact["missing_indices"]],
        "linear_parameters": len(artifact["coef"]) + 1, "feature_configuration": artifact["feature_config"],
        "model_configuration": artifact["metadata"]["classifier_hyperparameters"],
        "calibration": artifact["calibration"], "calibration_selection": artifact["metadata"]["calibration_selection"],
        "threshold": artifact["threshold"], "decision_policy_version": DECISION_POLICY_VERSION,
        "decision_policy": {"available": "p >= 0.5; confidence = probability of chosen class",
                            "insufficient": "quality <=0 or event_count <2 => internal abstention; /detect false with confidence 0.5, X-Behavior-Evidence=insufficient_evidence",
                            "fusion": "Consume /behavior quality; absent Behaviour is no positive synthetic vote, non-flag is not evidence of humanity"},
        "validation": checks["validation"], "official_train_calls": 282, "official_validation_calls": 71,
        "full_call_fallback_cases": 0, "historical_metric_files_unchanged": True,
        "evidence_files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in metrics_files},
        "runtime_revision_sha256": runtime_revision, "runtime_files_sha256": runtime_hashes,
        "source_revision_sha256": source_revision, "source_files_sha256": source_hashes,
        "changed_runtime_files_since_baseline": changed_runtime,
        "git_commit": None, "git_reason": "No project-level Git repository/commit; project is untracked in parent home repository; no commit/push performed",
        "vad_path": "artifacts/vad/silero_vad.onnx", "vad_sha256": sha(ROOT / "artifacts/vad/silero_vad.onnx"),
        "vad_configuration": artifact["metadata"]["vad_config"],
        "test_suite": tests, "runtime_smoke": isolated, "http_smoke": http,
        "runtime_python": isolated["python"], "runtime_packages": runtime_versions,
        "latency_existing_20_calls": latency["components_ms"]["total"],
        "peak_rss_mib_existing_benchmark": latency["peak_process_rss_mib"],
        "latency_scope": "Existing offline CPU benchmark; not rerun in delta review; excludes HTTP/network; post-fix smoke is not a new latency distribution",
        "docker_status": "NOT VERIFIED LOCALLY", "docker_review": "Static inspection only; Docker not installed",
        "permissions": "No external transfers/cloud/API credentials/remote Git changes; no confidential original data modified",
        "remaining_blockers": [],
        "limitations": ["71 development-validation calls with irreversible prior selection exposure",
                        "No verifiable source/session/engine grouping or pristine train-only selection winner",
                        "Rates, missingness and agent/protocol shortcuts remain",
                        "Selective non-overlap response population; automatic event boundaries, no causal/manual truth",
                        "Class-dependent VAD agreement introduces acoustic dependence",
                        "Weak short prefixes; fast computation is not early observation evidence",
                        "Train-OOF parameter calibration, validation-selected option; sparse bins and prior shift",
                        "Advanced autonomous agents may imitate timing; human-operated voice conversion not established",
                        "Limited robustness panel; no Docker/cloud/hidden-test/production proof"],
    }
    write_json(manifest_path, record)
    manifest_sha = sha(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(manifest_sha + "  FINAL_BEHAVIOR_MANIFEST.json\n")
    m = record["validation"]
    freeze = f'''# Final Behaviour integration freeze

**READY FOR INTEGRATION: YES** — engineering-ready for team integration under the
documented development-validation evidence. **Not proven robust on Altur's hidden
set or production banking traffic.** No further model optimization is authorized.

- Freeze timestamp (UTC): `{now}`.
- Trained artifact: `{record['artifact_path']}` (**unchanged** during delta review).
- Artifact SHA-256: `{record['artifact_sha256']}`.
- Artifact training timestamp: `{record['artifact_created_utc']}`.
- VAD: `{record['vad_path']}`; SHA-256 `{record['vad_sha256']}`.
- Runtime policy: `{DECISION_POLICY_VERSION}`.
- Runtime revision SHA-256: `{runtime_revision}`.
- Full source/test revision SHA-256: `{source_revision}`.
- Git commit: unavailable (project has no own `.git`; no commits/pushes made).
- Reference baseline: `{record['baseline_record']}` (read-only, 102 files).

## Exact model and metrics

- **24 primary features +15 missing indicators**; 40 base linear parameters including
  intercept, plus two sigmoid parameters. Full feature list and configuration:
  [FINAL_BEHAVIOR_MANIFEST.json](FINAL_BEHAVIOR_MANIFEST.json), `feature_names`,
  `missingness_indicators`, `feature_configuration`, `model_configuration`.
- L2 logistic C=0.1; train-only preprocessing; count-free primary feature candidate.
- Calibration: `{artifact['calibration']['method']}`, seed 2026;
  `sigmoid({artifact['calibration']['slope']:.16g} * logit + {artifact['calibration']['intercept']:.16g})`.
  Parameters fitted using training OOF logits/labels; the calibration option and
  count-free variant were selected with official validation involvement.
- Official train/val: **282 /71**. Accuracy **{m['accuracy']:.10f} (69/71)**;
  balanced accuracy **{m['balanced_accuracy']:.10f}**; ROC-AUC **{m['roc_auc']:.10f}**;
  FPR **1/37={m['fpr']:.10f}**; FNR **1/34={m['fnr']:.10f}**;
  Brier **{m['brier']:.10f}**; confusion **[[36,1],[1,33]]** (human, synthetic).
  Same artifact/hash and cached actual-WAV predictions; zero full-call fallback cases.

## Decision and fusion contract

Use `BehaviorDetector.predict(wav_bytes)` or `/behavior` and consume all four fields:
`synthetic_probability`, `quality_score`, `event_count`, `behavior_confidence`.
No manifest, labels, organizer JSON or training caches are required by inference.

For available evidence the demo `/detect` uses **p>=0.5**, confidence p or 1−p.
For q<=0 or fewer than two events it abstains internally and returns the Boolean-only
non-flag **false, confidence=0.5**, with `X-Behavior-Evidence: insufficient_evidence`.
This is **not evidence of humanity or access approval**. Fusion must give absent
Behaviour no positive synthetic vote; other low quality can be downweighted. No
fusion weights were invented. `behavior_confidence=q*abs(2p−1)` is a heuristic,
not probability of correctness. Debug diagnostics expose `evidence_state`.

## Verification

- Final complete suite run **once** in this delta review: **{tests['passed']} passed,
  {tests['skipped']} skipped, {tests['failures']} failures, {tests['errors']} errors**.
  Receipt: [final_tests.json](metrics/delta_review/final_tests.json) and
  [JUnit output](metrics/delta_review/final_pytest.xml). Earlier targeted regression
  exposed/fixed malformed RIFF handling, then all 12 targeted tests passed.
- Real local WAV in isolated data-free package: **PASS**, Python {isolated['python']},
  existing `.venv-runtime`; original project/data reads **OS-denied** and all network
  egress **OS-denied**. No sklearn/pandas/torch. Silence, sparse burst and one-channel
  audio abstain; repeated different calls preserve VAD state isolation.
- Actual loopback HTTP `/health`, `/behavior`, `/detect`, silence fallback/header
  and sanitized invalid input: **PASS**. Receipts: `metrics/delta_review/isolated_runtime.json`
  and `metrics/delta_review/http_smoke.json`.
- CPU benchmark retained (20 full calls): p50 **860.446 ms**, p95 **1193.124 ms**,
  peak **125.441 MiB**. Excludes HTTP/network and is not a new post-fix benchmark.
- Runtime dependencies are pinned in `requirements-runtime.lock`; installed package
  versions are recorded in the JSON manifest. No installation or cloud resource was needed.
- **Docker build/run: NOT VERIFIED LOCALLY**. Static Dockerfile/build-context review only.

## Remaining limitations and changes

The delta review fixed only the insufficient-evidence adapter and malformed RIFF
exception path, plus debug evidence-state/docs/reproducibility receipts. **No trained
artifact, VAD settings, selected features, calibration parameter or threshold changed.**

Prior validation selection cannot be undone; 71 calls do not establish banking error
rates. Group/engine/session provenance, causal interaction mechanisms and manual
speech truth remain unverified. Rates/missingness/protocol signal and acoustic VAD
dependence remain. Early prefixes are weak, robustness probes limited, calibration
domain-specific. Advanced agents can imitate timing; human-controlled voice conversion
is outside demonstrated autonomous-caller coverage. Docker/cloud reachability unverified.

See [FINAL_DELTA_REVIEW.md](FINAL_DELTA_REVIEW.md) for all ASTRA2 issue statuses,
exact evidence, diagnostic results, corrected cached-prefix policy tables and changes.

Manifest SHA-256: `{manifest_sha}`. Verify the model/runtime hashes before integration.
Freeze complete. **Stop here; no further model optimization.**
'''
    (ROOT / "reports/FINAL_BEHAVIOR_FREEZE.md").write_text(freeze)
    assert sha(artifact_path) == checks["artifact_sha256"]
    # Resolve local documentation links after writing the freeze/manifest.
    for relative in ("README.md", "behavior/README.md", "reports/BEHAVIOUR_REPORT.md",
                     "reports/FINAL_BEHAVIOR_FREEZE.md", "reports/FINAL_DELTA_REVIEW.md"):
        path = ROOT / relative
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), f"Broken local link in {relative}"
    print(json.dumps({"ready_for_integration": True, "artifact_changed": False,
                      "artifact_sha256": checks["artifact_sha256"], "test_suite": totals,
                      "runtime_revision_sha256": runtime_revision, "manifest_sha256": manifest_sha,
                      "docker_status": "NOT VERIFIED LOCALLY", "freeze_record": "reports/FINAL_BEHAVIOR_FREEZE.md"}, indent=2))


if __name__ == "__main__":
    main()
