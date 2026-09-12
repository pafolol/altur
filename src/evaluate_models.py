"""
THE EVALUATION MATRIX - Model A against Model B, on every domain we have.

    MODELS  (four, so the matrix can attribute an improvement rather than merely display it)
      specialist          Wav2Vec2 Spanish layer 5 + MLP, energy VAD. Exactly what is deployed today.
      specialist_silero   the same weights with the VAD swapped at INFERENCE only - the cheap partial fix.
      orig_only           original training data, Silero VAD, MLP re-fitted on Silero chunks. The segmentation
                          fix done properly; the only difference from robust_v2 is the training data.
      robust_v2           the same frozen backbone, a classifier trained on original + telephone-channel TRAIN.

    DOMAINS
      altur_val             the official validation split, untouched                    (saturated, 100 %)
      channel_val_seen      the same calls through G.711 / G.726 lines                  (codecs V2 trained on)
      channel_val_unseen    the same calls through GSM / AMR / iLBC / Speex / Opus / G.722 lines
                                                                                        (codecs NOBODY trained on)
      external_ood          new TTS engines and new speakers, off-pipeline
      external_ood_channel  the same clips through unseen-family telephone lines
      stress                the official validation calls under the 11 perturbations of stress_test.py

WHY THE MODELS SHARE ONE BACKBONE LOAD
  They can: same Hugging Face weights, possibly different layers. So the audio is segmented once per distinct
  voice-activity detector, the frozen backbone runs once per chunk set, and each model's MLP reads the layer
  it wants out of the same hidden states. Otherwise this script would run the backbone four times over
  everything for no reason.

AUC VERSUS THRESHOLD
  When accuracy collapses there are two very different explanations and they need different fixes:
      the score still RANKS the calls correctly (AUC stays high) and only the 0.5 cut is in the wrong place
          -> recalibrate, do not retrain
      the score stops separating the classes at all (AUC collapses)
          -> the representation is gone; only new training data fixes it
  For every model and domain this script reports the accuracy at the deployed threshold AND the accuracy at
  the best threshold that domain would allow, so the gap between them says which case we are in.

Run:  python src/evaluate_models.py [--domains all] [--skip-stress]
Outputs: outputs/robust/evaluation_matrix.{csv,json}, outputs/robust/scores_<model>_<domain>.csv,
         outputs/figures/evaluation_matrix.png, outputs/figures/auc_vs_threshold.png
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset
import phone_channel as pc
from build_channel_dataset import MANIFESTS
from metrics import aggregate_chunk_scores, evaluate_scores, log_experiment, save_json, sigmoid

RESULTS = config.OUTPUTS_DIR / "robust"
# Four configurations, chosen so the matrix can attribute any improvement to a cause rather than just show it:
#   specialist         what is deployed today: energy VAD, MLP fitted on energy-VAD chunks, original data
#   specialist_silero  the same weights with the VAD swapped at INFERENCE only - the cheap fix, and a partial
#                      one, because that MLP was fitted on chunks a different detector chose
#   orig_only          original data, Silero VAD, MLP re-fitted on Silero chunks - the segmentation fix done
#                      properly, with no new training data. The difference from robust_v2 is the data alone.
#   robust_v2          original + telephone-channel training data, Silero VAD
MODELS = {
    "specialist":        {"dir": "wav2vec2_spanish",  "vad": "energy", "label": "Specialist (deployed)"},
    "specialist_silero": {"dir": "wav2vec2_spanish",  "vad": "silero", "label": "Specialist, VAD swapped"},
    "orig_only":         {"dir": "robust_orig_only",  "vad": None,     "label": "Retrained, original data only"},
    "robust_v2":         {"dir": "robust_v2",         "vad": None,     "label": "Robust Acoustic V2"},
}
DOMAIN_ORDER = ["altur_val", "channel_val_seen", "channel_val_unseen",
                "external_ood", "external_ood_channel", "stress"]
# Every domain except external_ood is telephone audio - 8 kHz, band-limited, through a line. external_ood is
# clean studio and TTS material merely resampled to 8 kHz, which is NOT what this product ingests: the demo
# page already pushes such audio through a simulated line before scoring it, and a bank's call never arrives
# that way. It is reported in full, and it is the domain where the two models most disagree, so the deployment
# recommendation quotes BOTH the worst-over-everything and the worst-over-telephone numbers rather than
# choosing whichever flatters the newer model.
TELEPHONE_DOMAINS = [d for d in DOMAIN_ORDER if d != "external_ood"]


class Bench:
    """Several models that share one frozen backbone, scoring the same audio without redundant work."""

    def __init__(self, specs, device=None):
        from predict import AcousticDetector
        self.device = device or config.get_device(verbose=False)
        self.models = {}
        for name, spec in specs.items():
            det = AcousticDetector(spec["dir"], vad=spec["vad"], device=self.device, verbose=False)
            self.models[name] = det
            print(f"  {name:<18} {det.display}, layer {det.layer}, VAD {det.vad}, "
                  f"calibration {det.calibration.get('method', 'none')}")
        hf_names = {d.hf_name for d in self.models.values()}
        assert len(hf_names) == 1, f"Bench needs one shared backbone, got {hf_names}"
        self.max_layer = max(d.layer for d in self.models.values())
        self.vads = sorted({d.vad for d in self.models.values()})
        # one backbone, truncated at the deepest layer anybody asks for
        self.backbone = max(self.models.values(), key=lambda d: d.layer)
        self.do_normalize = self.backbone.do_normalize

    @torch.inference_mode()
    def _hidden(self, chunks):
        """list of 16 kHz chunks -> {layer: (n_chunks, dim) tensor} for every layer any model needs."""
        wanted = sorted({d.layer for d in self.models.values()})
        out = {layer: [] for layer in wanted}
        for c in chunks:
            x = torch.from_numpy(audio.normalize_for_model(c, self.do_normalize))[None].to(self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                hs = self.backbone.model(x, output_hidden_states=True).hidden_states
            for layer in wanted:
                out[layer].append(hs[layer].float().mean(dim=1))
        return {layer: torch.cat(v) for layer, v in out.items()}

    def score(self, stereo, sr):
        """(n, ch) array -> {model name: dict with score, probability, n_chunks}."""
        results = {}
        for vad in self.vads:
            chunks, spans = audio.caller_chunks(stereo, sr, vad_method=vad)
            users = [n for n, d in self.models.items() if d.vad == vad]
            if not chunks:
                for n in users:
                    results[n] = {"score": np.nan, "p": 0.5, "n_chunks": 0, "no_speech": True}
                continue
            hidden = self._hidden(chunks)
            for n in users:
                det = self.models[n]
                chunk_scores = np.asarray(det.chunk_scores(hidden[det.layer]), dtype=float)
                score = aggregate_chunk_scores(chunk_scores, det.aggregation)
                results[n] = {"score": float(score), "p": det.calibrate(score),
                              "n_chunks": len(chunks), "no_speech": False}
        return results


# ----------------------------------------------------------------------------- the domains
def domain_altur_val():
    df = dataset.load_split("val")
    return [(r.anon_id, r.y, r.path, None) for r in df.itertuples()]


def domain_channel(family):
    man = pd.read_csv(MANIFESTS / "all.csv")
    man = man[(man["status"] == "ok") & (man["split"] == "val") & (man["family"] == family)]
    if man.empty:
        return []
    return [(r.sample_id, int(r.y), r.channel_path, None) for r in man.itertuples()]


def domain_external(through_channel):
    """
    The out-of-dataset clips built by external_check.py (new TTS engines, new speakers). `through_channel`
    pushes each of them through a freshly drawn UNSEEN-family telephone line, which is a genuine double
    generalisation test: engines nobody trained on, down a codec nobody trained on.
    """
    clips = sorted((config.OUTPUTS_DIR / "external" / "clips").glob("call_*.wav"))
    items = []
    for p in clips:
        label = p.stem.split("_")[2]
        items.append((p.stem, int(label == "synthetic"), str(p), "unseen" if through_channel else None))
    return items


def domain_stress():
    import stress_test
    df = dataset.load_split("val")
    return [(f"{r.anon_id}|{cond}", r.y, r.path, ("stress", cond))
            for r in df.itertuples() for cond in stress_test.CONDITIONS]


def load_audio(path, transform, item_id):
    """Read a file and apply the domain's on-the-fly transform to the caller channel, if any."""
    wave, sr = audio.read_wav(path)
    if transform is None:
        return wave, sr
    if isinstance(transform, tuple) and transform[0] == "stress":
        import stress_test
        wave = wave.copy()
        wave[:, config.CALLER_CHANNEL] = stress_test.CONDITIONS[transform[1]](wave[:, config.CALLER_CHANNEL])
        return wave, sr
    # a freshly drawn telephone channel of the named family, seeded by the item id so it is reproducible
    seed = int.from_bytes(__import__("hashlib").sha256(f"{item_id}|{transform}".encode()).digest()[:8], "big")
    caller = audio.get_channel(wave, config.CALLER_CHANNEL)
    regions = audio.detect_speech_energy(caller, sr)
    mask = pc.speech_mask_from_regions(regions, len(caller))
    params = pc.sample_channel(np.random.default_rng(seed), family=transform)
    out, _ = pc.apply_channel(caller, params, mask, seed=seed % (2 ** 31))
    wave = wave.copy()
    wave[:, config.CALLER_CHANNEL] = out
    return wave, sr


def run_domain(bench, name, items):
    rows = []
    for item_id, y, path, transform in tqdm(items, desc=f"[{name}]", unit="file", leave=False):
        wave, sr = load_audio(path, transform, item_id)
        got = bench.score(wave, sr)
        for model, r in got.items():
            rows.append({"domain": name, "model": model, "item": item_id, "y": int(y), **r})
    return pd.DataFrame(rows)


def summarise(df):
    """Per (domain, model): the deployed-threshold metrics plus what the best threshold would have given."""
    out = []
    for (domain, model), g in df.groupby(["domain", "model"]):
        # A file with no detected speech is a real production outcome: the endpoint answers 0.5, which the
        # challenge contract reads as "synthetic". Scoring it any other way would flatter the model.
        scores = g["score"].to_numpy(dtype=float)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        y = g["y"].to_numpy()
        m = evaluate_scores(y, scores)
        best_acc, best_thr = max(((float(((scores >= t).astype(int) == y).mean()), float(t))
                                  for t in np.unique(np.concatenate([scores, [0.0]]))), key=lambda z: z[0])
        out.append({"domain": domain, "model": model, "n": int(len(g)),
                    "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"], "f1": m["f1"],
                    "auc": m["auc"], "eer": m["eer"], "far": m["far"], "frr": m["frr"], "brier": m["brier"],
                    "accuracy_best_threshold": best_acc, "best_threshold": best_thr,
                    "threshold_headroom": best_acc - m["accuracy"],
                    "files_without_speech": int(g["no_speech"].sum())})
    return pd.DataFrame(out)


COLOURS = {"specialist": "#7f7f7f", "specialist_silero": "#9edae5", "orig_only": "#1f77b4", "robust_v2": "#d62728"}


def plot_matrix(summary, path):
    """One panel per metric, stacked - with four models and six domains a side-by-side layout is unreadable
    once the figure is scaled down to the 17 cm of a printed page."""
    domains = [d for d in DOMAIN_ORDER if d in set(summary["domain"])]
    models = [m for m in MODELS if m in set(summary["model"])]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    width = 0.8 / len(models)
    for ax, metric, title in ((axes[0], "accuracy", "call accuracy at the deployed threshold"),
                              (axes[1], "auc", "ROC AUC (ranking quality, threshold-free)")):
        x = np.arange(len(domains))
        for i, model in enumerate(models):
            vals = [float(summary[(summary["domain"] == d) & (summary["model"] == model)][metric].iloc[0])
                    if len(summary[(summary["domain"] == d) & (summary["model"] == model)]) else np.nan
                    for d in domains]
            pos = x + i * width - 0.4 + width / 2
            ax.bar(pos, vals, width, label=MODELS[model]["label"], color=COLOURS[model])
            for xi, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, [d.replace("_", "\n") for d in domains], fontsize=10)
        ax.set_ylim(0.3, 1.10)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=10, loc="lower left", ncol=2)
    fig.suptitle("Specialist vs Robust Acoustic V2 (dotted line = chance)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_auc_vs_threshold(summary, path):
    """Is a collapse a calibration problem or a representation problem? This is the picture that answers it."""
    domains = [d for d in DOMAIN_ORDER if d in set(summary["domain"])]
    models = [m for m in MODELS if m in set(summary["model"])]
    ncol = 2 if len(models) > 2 else len(models)
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.8 * nrow), squeeze=False)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax, model in zip(axes.ravel(), models):
        sub = summary[summary["model"] == model].set_index("domain")
        x = np.arange(len(domains))
        acc = [sub.loc[d, "accuracy"] for d in domains]
        best = [sub.loc[d, "accuracy_best_threshold"] for d in domains]
        auc = [sub.loc[d, "auc"] for d in domains]
        ax.bar(x, acc, 0.55, color="#d62728", label="accuracy at the deployed threshold")
        ax.bar(x, np.array(best) - np.array(acc), 0.55, bottom=acc, color="#ffbb78",
               label="what recalibration alone would recover")
        ax.plot(x, auc, "ko-", ms=5, label="AUC (what the representation still knows)")
        ax.set_xticks(x, [d.replace("_", "\n") for d in domains], fontsize=8.5)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_title(MODELS[model]["label"], fontsize=12)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Calibration shift or representation collapse? A tall orange band means the scores still "
                 "rank correctly and only the threshold moved", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["all"], choices=["all", *DOMAIN_ORDER])
    parser.add_argument("--models", nargs="+", default=["all"], choices=["all", *MODELS])
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--from-cache", action="store_true",
                        help="rebuild the summary, tables and figures from the per-domain score CSVs")
    args = parser.parse_args()
    config.set_seed()
    RESULTS.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    specs = {k: v for k, v in MODELS.items() if "all" in args.models or k in args.models}
    specs = {k: v for k, v in specs.items() if (config.MODELS_DIR / v["dir"] / "meta.json").exists()}
    print("[models]")
    bench = None if args.from_cache else Bench(specs)

    builders = {
        "altur_val": domain_altur_val,
        "channel_val_seen": lambda: domain_channel("seen"),
        "channel_val_unseen": lambda: domain_channel("unseen"),
        "external_ood": lambda: domain_external(False),
        "external_ood_channel": lambda: domain_external(True),
        "stress": domain_stress,
    }
    names = DOMAIN_ORDER if "all" in args.domains else args.domains
    if args.skip_stress:
        names = [n for n in names if n != "stress"]

    frames, t0 = [], time.time()
    for name in names:
        items = builders[name]()
        if not items:
            print(f"[{name}] no items -> skipped")
            continue
        if args.limit:
            items = items[:args.limit]
        if args.from_cache:
            cached = RESULTS / f"scores_{name}.csv"
            if not cached.exists():
                print(f"[{name}] no cached scores -> skipped")
                continue
            print(f"[{name}] reading cached scores")
            frames.append(pd.read_csv(cached))
            continue
        print(f"[{name}] {len(items)} files x {len(specs)} models")
        df = run_domain(bench, name, items)
        df.to_csv(RESULTS / f"scores_{name}.csv", index=False)
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True)
    summary = summarise(scores)
    summary.to_csv(RESULTS / "evaluation_matrix.csv", index=False)

    # ------------------------------------------------------------------ the table
    print("\n" + "=" * 118)
    head = f"{'domain':<22}" + "".join(f"{MODELS[m]['label']:>32}" for m in specs)
    print(head)
    print("-" * 118)
    for d in [x for x in DOMAIN_ORDER if x in set(summary['domain'])]:
        line = f"{d:<22}"
        for m in specs:
            r = summary[(summary["domain"] == d) & (summary["model"] == m)]
            if r.empty:
                line += f"{'-':>32}"
            else:
                r = r.iloc[0]
                line += f"{r['accuracy']:>12.3f} acc /{r['auc']:>7.4f} AUC"
        print(line)
    print("=" * 118)
    worst = {m: float(summary[summary["model"] == m]["accuracy"].min()) for m in specs}
    phone = summary[summary["domain"].isin(TELEPHONE_DOMAINS)]
    worst_phone = {m: float(phone[phone["model"] == m]["accuracy"].min()) for m in specs}
    print("worst accuracy over ALL domains:        " + "   ".join(f"{MODELS[m]['label']}: {v:.3f}" for m, v in worst.items()))
    print("worst accuracy over TELEPHONE domains:  " + "   ".join(f"{MODELS[m]['label']}: {v:.3f}" for m, v in worst_phone.items()))
    best_all, best_model = max(worst, key=worst.get), max(worst_phone, key=worst_phone.get)
    print(f"-> best by worst-over-all:       {MODELS[best_all]['label']}")
    print(f"-> best by worst-over-telephone: {MODELS[best_model]['label']}   (the deployment criterion)")
    if best_all != best_model:
        print("   The two rules disagree. external_ood is clean non-telephone audio; see the report, section 8.")

    plot_matrix(summary, config.FIGURES_DIR / "evaluation_matrix.png")
    plot_auc_vs_threshold(summary, config.FIGURES_DIR / "auc_vs_threshold.png")
    save_json({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": time.time() - t0,
               "models": {k: (v if bench is None else {**v, "layer": bench.models[k].layer, "vad": bench.models[k].vad})
                          for k, v in specs.items()},
               "worst_domain_accuracy": worst, "worst_telephone_accuracy": worst_phone,
               "telephone_domains": TELEPHONE_DOMAINS,
               "best_model": best_model, "best_model_all_domains": best_all,
               "matrix": summary.to_dict(orient="records")}, RESULTS / "evaluation_matrix.json")
    # One pointer file, written from the measurement, so predict.py (the fusion interface) and server.py
    # (the HTTP endpoint) cannot drift apart about which model is deployed. Reverting is editing one line.
    save_json({"model": best_model, "dir": specs[best_model]["dir"], "label": MODELS[best_model]["label"],
               "criterion": "highest worst-case accuracy over the telephone domains "
                            f"({', '.join(TELEPHONE_DOMAINS)})",
               "worst_telephone_accuracy": worst_phone[best_model],
               "worst_all_domains_accuracy": worst[best_model],
               "runner_up_all_domains": best_all,
               "note": "external_ood (clean non-telephone audio) is excluded from the criterion and reported "
                       "in full in the report; see section 8.",
               "generated": time.strftime("%Y-%m-%d %H:%M:%S")}, config.REPORTS_DIR / "deployed_model.json")
    log_experiment(title="evaluation matrix - specialist vs robust v2",
                   models=", ".join(specs), domains=", ".join(names),
                   runtime_s=round(time.time() - t0, 1),
                   worst_domain_accuracy={k: round(v, 4) for k, v in worst.items()},
                   worst_telephone_accuracy={k: round(v, 4) for k, v in worst_phone.items()},
                   best_model=best_model, best_model_all_domains=best_all,
                   notes="channel_val_unseen never used for training or selection; external_ood is clean "
                         "non-telephone audio and is excluded from the deployment criterion but reported in full")
    print(f"\ndeployed -> {config.REPORTS_DIR / 'deployed_model.json'} ({MODELS[best_model]['label']})")
    print(f"matrix  -> {RESULTS / 'evaluation_matrix.csv'}")
    print(f"figures -> {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
