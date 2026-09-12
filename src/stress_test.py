"""
ROBUSTNESS STRESS TEST - because the clean validation split is saturated.

Every frozen backbone (and every layer) separates human from synthetic callers perfectly on the official
validation split, and a model on trivial statistics reaches AUC 1.0. Accuracy on clean validation therefore
cannot rank the backbones, and it tells us nothing about the judges' first criterion: robustness to unseen
call conditions. So we re-score the SAME validation calls after perturbing the caller channel in ways the
training data never contained (test-time only, nothing is retrained):

    gain -12 dB            sanity check: loudness normalisation should make this a no-op
    lowpass 3.4 kHz        removes the 3.3 kHz band edge that distinguishes synthetic calls -> does a human call flip?
    lowpass 3.0 kHz        stronger band limit
    pstn 300-3400 Hz       classic telephone band-pass
    white noise 20 / 10 dB additive noise at a given SNR (relative to the caller's speech level)
    pink noise 10 dB       speech-like coloured noise
    mu-law codec           G.711 8-bit companding (real telephone codec)
    reverb 0.3 s           small-room reverberation (synthetic exponentially decaying impulse response)
    spectral tilt -3 dB/oct a different handset / microphone response

For each backbone the validation embeddings of EVERY layer are extracted under every condition and cached,
then the layer probe (logistic regression trained on clean train chunks) and the MLP are evaluated per
condition. The ROBUST layer of a backbone = the layer with the highest mean validation AUC over all
conditions (clean included); the ROBUST score of a model = its mean AUC over the conditions.

Run:  python src/stress_test.py --backbone all --stage extract      (val embeddings under every condition; GPU)
      python src/stress_test.py --backbone all --stage probe        (per-layer robustness -> robust layer)
      python src/stress_test.py --backbone all --stage mlp          (after train_classifier: MLP under stress)
Outputs: outputs/embeddings/<backbone>/stress/<condition>_val_{mean,index}.*,
         outputs/stress/<backbone>/{probe_stress.json, mlp_stress.json, probe_heatmap.png}, outputs/figures/stress_*.png
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, lfilter, sosfiltfilt
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from metrics import evaluate_scores, log_experiment, save_json

SR = config.SOURCE_SAMPLE_RATE


# ----------------------------------------------------------------------------- perturbations (8 kHz caller channel)
def _speech_rms(wave):
    regions = audio.detect_speech_energy(wave, SR)
    speech = np.concatenate([audio.slice_chunk(wave, SR, s, e) for s, e in regions]) if regions else wave
    return np.sqrt(np.mean(speech ** 2)) + 1e-9


def add_noise(wave, snr_db, pink=False, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(wave)).astype(np.float32)
    if pink:  # 1/f spectrum via FFT shaping
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1 / SR)
        spec[1:] /= np.sqrt(freqs[1:])
        noise = np.fft.irfft(spec, n=len(noise)).astype(np.float32)
    noise *= _speech_rms(wave) / (np.sqrt(np.mean(noise ** 2)) + 1e-9) / (10 ** (snr_db / 20))
    return np.clip(wave + noise, -1, 1).astype(np.float32)


def lowpass(wave, cutoff, order=8):
    sos = butter(order, cutoff / (SR / 2), btype="low", output="sos")
    return sosfiltfilt(sos, wave).astype(np.float32)


def bandpass(wave, lo, hi, order=6):
    sos = butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, wave).astype(np.float32)


def mulaw(wave, mu=255.0):
    x = np.clip(wave, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * 255) / 255 * 2 - 1  # 8-bit quantisation
    return (np.sign(q) * (1 / mu) * ((1 + mu) ** np.abs(q) - 1)).astype(np.float32)


def reverb(wave, rt60=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2))
    out = np.convolve(wave, rir)[:len(wave)]
    return (out * (np.sqrt(np.mean(wave ** 2)) / (np.sqrt(np.mean(out ** 2)) + 1e-9))).astype(np.float32)


def tilt(wave, db_per_octave=-3.0):
    spec = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), 1 / SR)
    gain = np.ones_like(freqs)
    gain[1:] = 10 ** (db_per_octave * np.log2(freqs[1:] / 1000.0) / 20)
    return np.fft.irfft(spec * gain, n=len(wave)).astype(np.float32)


CONDITIONS = {
    "clean": lambda w: w,
    "gain_-12dB": lambda w: (w * 10 ** (-12 / 20)).astype(np.float32),
    "lowpass_3400": lambda w: lowpass(w, 3400),
    "lowpass_3000": lambda w: lowpass(w, 3000),
    "pstn_300-3400": lambda w: bandpass(w, 300, 3400),
    "white_snr20": lambda w: add_noise(w, 20),
    "white_snr10": lambda w: add_noise(w, 10),
    "pink_snr10": lambda w: add_noise(w, 10, pink=True),
    "mulaw_codec": mulaw,
    "reverb_0.3s": reverb,
    "tilt_-3dB_oct": tilt,
}
STRESSED = [c for c in CONDITIONS if c != "clean"]


# ----------------------------------------------------------------------------- extraction
def extract(backbone, conditions, device, force=False):
    from extract_embeddings import embed_chunks, load_backbone
    out = config.EMBEDDINGS_DIR / backbone / "stress"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in conditions if force or not (out / f"{c}_val_mean.npy").exists()]
    if not todo:
        print(f"[{backbone}] stress embeddings cached")
        return
    model, do_normalize, _ = load_backbone(backbone, device)
    df = dataset.load_split("val")
    for cond in todo:
        t0 = time.time()
        means, index = [], []
        for row in tqdm(df.itertuples(), total=len(df), desc=f"[{backbone}] {cond}", unit="call", leave=False):
            stereo, sr = audio.read_wav(row.path)
            stereo = stereo.copy()
            stereo[:, config.CALLER_CHANNEL] = CONDITIONS[cond](stereo[:, config.CALLER_CHANNEL])
            chunks, spans = audio.caller_chunks(stereo, sr)
            if not chunks:
                continue
            m, _ = embed_chunks(model, chunks, do_normalize, device)
            means.append(m.astype(np.float16))
            index += [{"anon_id": row.anon_id, "y": row.y, "chunk": k, "start_s": s, "end_s": e, "duration_s": e - s}
                      for k, (s, e) in enumerate(spans)]
        np.save(out / f"{cond}_val_mean.npy", np.concatenate(means))
        pd.DataFrame(index).to_csv(out / f"{cond}_val_index.csv", index=False)
        print(f"[{backbone}] {cond}: {len(index)} chunks in {time.time() - t0:.0f}s")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_stress(backbone, cond):
    out = config.EMBEDDINGS_DIR / backbone / "stress"
    return pd.read_csv(out / f"{cond}_val_index.csv"), np.load(out / f"{cond}_val_mean.npy").astype(np.float32)


_VAL = None


def complete_calls(calls):
    """
    A perturbation can leave a call without any detected speech (e.g. heavy noise defeats the VAD). Such a call
    is not dropped from the metric: it gets score 0 (p = 0.5, undecided), exactly what the endpoint would return.
    """
    global _VAL
    if _VAL is None:
        _VAL = dataset.load_split("val")[["anon_id", "y"]]
    missing = _VAL[~_VAL["anon_id"].isin(calls["anon_id"])]
    if len(missing):
        extra = pd.DataFrame({"anon_id": missing["anon_id"].to_numpy(), "y": missing["y"].to_numpy(), "score": 0.0,
                              "n_chunks": 0, "chunk_scores": [[] for _ in range(len(missing))]})
        calls = pd.concat([calls, extra], ignore_index=True).sort_values("anon_id").reset_index(drop=True)
    return calls


# ----------------------------------------------------------------------------- probe under stress
def probe(backbone, conditions):
    from train_classifier import call_table, load_embeddings, make_logreg
    idx_tr, E_tr = load_embeddings(backbone, "train")
    ytr = idx_tr["y"].to_numpy()
    stress = {c: load_stress(backbone, c) for c in conditions}
    n_layers = E_tr.shape[1]
    table = np.zeros((n_layers, len(conditions)))
    acc = np.zeros_like(table)
    eer = np.zeros_like(table)
    rows = []
    for layer in range(n_layers):
        clf = make_logreg().fit(E_tr[:, layer], ytr)
        for j, cond in enumerate(conditions):
            idx, E = stress[cond]
            calls = complete_calls(call_table(idx, clf.decision_function(E[:, layer])))
            m = evaluate_scores(calls["y"], calls["score"])
            table[layer, j], acc[layer, j], eer[layer, j] = m["auc"], m["accuracy"], m["eer"]
            rows.append({"layer": layer, "condition": cond, **{k: m[k] for k in ("auc", "accuracy", "balanced_accuracy", "eer", "f1", "far", "frr")}})
        mean_auc, mean_acc = table[layer].mean(), acc[layer].mean()
        print(f"  layer {layer:2d}: mean AUC {mean_auc:.4f}  mean acc {mean_acc:.3f}  worst AUC {table[layer].min():.3f} "
              f"({conditions[int(np.argmin(table[layer]))]})")
    stressed_cols = [j for j, c in enumerate(conditions) if c != "clean"]
    mean_auc = table.mean(axis=1)
    mean_eer = eer.mean(axis=1)
    order = sorted(range(n_layers), key=lambda l: (-round(mean_auc[l], 4), round(mean_eer[l], 4), l))
    robust_layer = order[0]
    res = {"backbone": backbone, "conditions": conditions, "rows": rows, "robust_layer": robust_layer,
           "mean_auc_per_layer": mean_auc.tolist(), "mean_acc_per_layer": acc.mean(axis=1).tolist(),
           "stressed_mean_auc_per_layer": table[:, stressed_cols].mean(axis=1).tolist(),
           "worst_auc_per_layer": table.min(axis=1).tolist(),
           "selection_rule": "highest mean validation AUC over clean + stressed conditions; ties -> lower mean EER -> lower layer"}
    out = config.OUTPUTS_DIR / "stress" / backbone
    save_json(res, out / "probe_stress.json")
    plot_heatmap(table, acc, conditions, robust_layer, backbone, out / "probe_heatmap.png")
    print(f"  -> robust layer {robust_layer} (mean AUC {mean_auc[robust_layer]:.4f}, worst {table[robust_layer].min():.3f})")
    log_experiment(title=f"stress-test layer probe - {backbone}", model=config.BACKBONES[backbone]["hf_name"], dataset_fraction=1.0,
                   segmentation="as pipeline; validation caller channel perturbed", selected_layer=robust_layer, classifier="logreg per layer (clean train)",
                   hyperparameters=f"conditions={conditions}", device="gpu (extraction) / cpu (probe)", runtime_s="n/a",
                   mean_auc=round(float(mean_auc[robust_layer]), 4), worst_auc=round(float(table[robust_layer].min()), 4),
                   notes="clean validation saturated (AUC 1.0 on every layer); layer chosen on stressed validation")
    return res


def plot_heatmap(auc, acc, conditions, robust_layer, backbone, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.32 * auc.shape[0] + 2.4))
    for ax, M, name in ((axes[0], auc, "ROC AUC"), (axes[1], acc, "accuracy")):
        im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(conditions)), conditions, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(M.shape[0]), [f"L{i}" + (" *" if i == robust_layer else "") for i in range(M.shape[0])], fontsize=7)
        ax.set_title(f"validation {name} per layer x condition (logistic-regression probe)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f"{config.BACKBONES[backbone]['display']}: robustness of every frozen layer (* = robust layer)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    import shutil
    shutil.copy(path, config.FIGURES_DIR / f"stress_probe_{backbone}.png")


# ----------------------------------------------------------------------------- MLP under stress
def mlp_stress(backbone, conditions, device):
    from train_classifier import MLP, call_table
    model_dir = config.MODELS_DIR / backbone
    meta = json.loads((model_dir / "meta.json").read_text())
    layer = meta["layer"]
    ckpt = torch.load(model_dir / "mlp.pt", map_location=device)
    mlp = MLP(ckpt["dim"], ckpt["hidden"], ckpt["dropout"]).to(device)
    mlp.load_state_dict(ckpt["state_dict"])
    import joblib
    logreg = joblib.load(model_dir / "logreg.joblib")
    res = {"backbone": backbone, "layer": layer, "conditions": {}}
    preds = []
    for cond in conditions:
        idx, E = load_stress(backbone, cond)
        X = E[:, layer]
        s_mlp = mlp.scores(torch.tensor(X, device=device))
        s_lr = logreg.decision_function(X)
        c_mlp, c_lr = complete_calls(call_table(idx, s_mlp)), complete_calls(call_table(idx, s_lr))
        res["conditions"][cond] = {"mlp": evaluate_scores(c_mlp["y"], c_mlp["score"]), "logreg": evaluate_scores(c_lr["y"], c_lr["score"]),
                                   "n_chunks": int(len(idx)), "calls_without_speech": int((c_mlp["n_chunks"] == 0).sum())}
        m = res["conditions"][cond]["mlp"]
        # how many calls flipped class vs clean?
        preds.append(c_mlp.assign(condition=cond)[["anon_id", "y", "score", "condition"]])
        print(f"  {cond:<16} MLP: acc {m['accuracy']:.3f}  bal_acc {m['balanced_accuracy']:.3f}  AUC {m['auc']:.4f}  EER {m['eer'] * 100:5.2f}%  "
              f"FAR {m['far']:.2f} FRR {m['frr']:.2f}   | logreg AUC {res['conditions'][cond]['logreg']['auc']:.4f}   "
              f"| {len(idx)} chunks, {res['conditions'][cond]['calls_without_speech']} calls without speech")
    stressed = [c for c in conditions if c != "clean"]
    for clf in ("mlp", "logreg"):
        res[f"{clf}_mean_auc_stressed"] = float(np.mean([res["conditions"][c][clf]["auc"] for c in stressed]))
        res[f"{clf}_mean_accuracy_stressed"] = float(np.mean([res["conditions"][c][clf]["accuracy"] for c in stressed]))
        res[f"{clf}_worst_auc"] = float(min(res["conditions"][c][clf]["auc"] for c in conditions))
        res[f"{clf}_worst_condition"] = min(conditions, key=lambda c: res["conditions"][c][clf]["auc"])
        res[f"{clf}_mean_auc_all"] = float(np.mean([res["conditions"][c][clf]["auc"] for c in conditions]))
    pd.concat(preds).to_csv(config.PREDICTIONS_DIR / f"{backbone}_stress_val.csv", index=False)
    out = config.OUTPUTS_DIR / "stress" / backbone
    save_json(res, out / "mlp_stress.json")
    print(f"  -> MLP mean stressed AUC {res['mlp_mean_auc_stressed']:.4f}, mean stressed accuracy {res['mlp_mean_accuracy_stressed']:.3f}, "
          f"worst {res['mlp_worst_auc']:.3f} ({res['mlp_worst_condition']})")
    log_experiment(title=f"stress-test MLP - {backbone}", model=config.BACKBONES[backbone]["hf_name"], dataset_fraction=1.0,
                   segmentation="validation caller channel perturbed", selected_layer=layer, classifier="MLP (clean train)",
                   hyperparameters=f"conditions={conditions}", device=str(device), runtime_s="seconds",
                   mean_stressed_auc=round(res["mlp_mean_auc_stressed"], 4), mean_stressed_acc=round(res["mlp_mean_accuracy_stressed"], 4),
                   worst=f"{res['mlp_worst_condition']} AUC {res['mlp_worst_auc']:.3f}")
    return res


def plot_mlp_stress(backbones):
    results = {k: json.loads((config.OUTPUTS_DIR / "stress" / k / "mlp_stress.json").read_text()) for k in backbones
               if (config.OUTPUTS_DIR / "stress" / k / "mlp_stress.json").exists()}
    if not results:
        return
    conds = list(next(iter(results.values()))["conditions"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    x = np.arange(len(conds))
    w = 0.8 / len(results)
    colors = {"wavlm": "#1f77b4", "wav2vec2_spanish": "#ff7f0e", "xlsr": "#2ca02c"}
    for ax, metric in zip(axes, ("auc", "accuracy")):
        for i, (k, r) in enumerate(results.items()):
            vals = [r["conditions"][c]["mlp"][metric] for c in conds]
            ax.bar(x + (i - (len(results) - 1) / 2) * w, vals, w, color=colors[k], label=f"{config.BACKBONES[k]['display']} (layer {r['layer']})")
        ax.set_xticks(x, conds, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.02)
        ax.axhline(1.0, color="gray", lw=0.6)
        ax.set_ylabel(f"validation {metric}")
        ax.set_title(f"MLP {metric} on validation under each perturbation of the caller channel", fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Robustness stress test: same validation calls, caller channel perturbed at test time only")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "stress_mlp.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all", choices=["all"] + config.BACKBONE_ORDER)
    parser.add_argument("--stage", default="all", choices=["all", "extract", "probe", "mlp", "plot"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config.set_seed()
    config.ensure_dirs()
    device = config.get_device()
    keys = config.BACKBONE_ORDER if args.backbone == "all" else [args.backbone]
    for key in keys:
        if args.stage in ("all", "extract"):
            extract(key, args.conditions, device, args.force)
        if args.stage in ("all", "probe"):
            print(f"\n[{key}] layer probe under stress")
            probe(key, args.conditions)
        if args.stage in ("all", "mlp"):
            print(f"\n[{key}] MLP under stress")
            mlp_stress(key, args.conditions, device)
    if args.stage in ("all", "mlp", "plot"):
        plot_mlp_stress(keys if args.backbone != "all" else config.BACKBONE_ORDER)


if __name__ == "__main__":
    main()
