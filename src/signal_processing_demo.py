"""
SIGNAL-PROCESSING WALKTHROUGH on real Altur calls: the figures for the "audio fundamentals" part of the report.

Picks one human and one synthetic call from the TRAIN split with comparable caller speech time, then draws:
  waveform of the whole call (both channels) / zoomed waveform / sampling & Nyquist / FFT of one frame
  spectrogram / mel spectrogram / MFCC / 8 kHz vs 16 kHz resampling / long-term average spectrum per class
  pitch (F0) statistics on a subset of calls / caller-agent turn structure

Everything is computed at the NATIVE 8 kHz (Nyquist 4 kHz) so the plots show what the telephone line
actually carries. Nothing here is a classifier; visual differences are illustrations, not proof.

Run:  python src/signal_processing_demo.py
Outputs: outputs/figures/sp_*.png, outputs/signal_processing/examples.json (+ example WAVs of the caller turns)
"""
import json

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft
import soundfile as sf

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from metrics import save_json

FIG = config.FIGURES_DIR
OUT = config.OUTPUTS_DIR / "signal_processing"
SR = config.SOURCE_SAMPLE_RATE
N_FFT, HOP, N_MELS, N_MFCC = 512, 80, 64, 20   # 64 ms window, 10 ms hop at 8 kHz
COLORS = {"human": "#1f77b4", "synthetic": "#d62728"}


def pick_examples():
    """One human + one synthetic train call, both close to the median caller speech time and VAD agreement > 0.9."""
    stats = pd.read_csv(config.OUTPUTS_DIR / "dataset_inspection" / "call_stats.csv")
    stats = stats[(stats["split"] == "train") & (stats["vad_f1"] > 0.9)]
    picks = {}
    for label in config.LABELS:
        d = stats[stats["label"] == label].copy()
        target = stats["caller_speech_s"].median()
        # closest to the class-typical call: speech time, loudness, noise floor and SNR near the class medians
        d = d[d["n_caller_turns"] >= 12] if (d["n_caller_turns"] >= 12).sum() >= 5 else d
        d["dist"] = ((d["caller_speech_s"] - target).abs() / 10 + (d["caller_speech_rms_db"] - d["caller_speech_rms_db"].median()).abs() / 3
                     + (d["caller_noise_floor_db"] - d["caller_noise_floor_db"].median()).abs() / 3
                     + (d["caller_snr_db"] - d["caller_snr_db"].median()).abs() / 3)
        picks[label] = d.sort_values("dist").iloc[0]
    return picks


def load_call(anon_id):
    stereo, sr = audio.read_wav(config.AUDIO_DIR / f"{anon_id}.wav")
    caller, agent = audio.get_channel(stereo, 0), audio.get_channel(stereo, 1)
    turns = dataset.load_turns(config.TURNS_DIR / f"{anon_id}.json")
    return caller, agent, sr, turns


def longest_turn(caller, sr, regions, max_s=4.0):
    s, e = max(regions, key=lambda r: r[1] - r[0])
    e = min(e, s + max_s)
    return audio.slice_chunk(caller, sr, s, e), s


def representations(wave):
    S = np.abs(librosa.stft(wave, n_fft=N_FFT, hop_length=HOP))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=N_MELS, fmax=SR / 2)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=N_MFCC)
    return S_db, mel_db, mfcc


# ----------------------------------------------------------------------------- figures
def fig_call_overview(examples):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    for ax, (label, ex) in zip(axes, examples.items()):
        caller, agent, sr, turns, regions = ex["caller"], ex["agent"], ex["sr"], ex["turns"], ex["regions"]
        t = np.arange(len(caller)) / sr
        ax.plot(t, agent + 1.2, lw=0.3, color="#2ca02c", label="channel 1 = agent (offset +1.2)")
        ax.plot(t, caller, lw=0.3, color=COLORS[label], label="channel 0 = caller")
        for s, e in regions:
            ax.axvspan(s, e, ymin=0.0, ymax=0.5, color=COLORS[label], alpha=0.12)
        for tr in dataset.turns_of_channel(turns, 0):
            ax.plot(tr, [-1.05, -1.05], color="black", lw=3, solid_capstyle="butt")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-1.15, 2.3)
        ax.set_yticks([0, 1.2], ["caller", "agent"])
        ax.set_title(f"{label.upper()} caller  [{ex['anon_id']}]  {len(caller) / sr:.0f} s, stereo 8 kHz - "
                     f"shaded: our VAD regions, black bars: official caller turns", fontsize=9.5)
        ax.legend(fontsize=7, loc="upper right")
    axes[1].set_xlabel("time (s)")
    fig.suptitle("A whole call: the caller and the agent live on separate channels; only the caller is classified")
    fig.tight_layout()
    fig.savefig(FIG / "sp_call_overview.png", dpi=130)
    plt.close(fig)


def fig_waveform_zoom(examples):
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    for row, (label, ex) in enumerate(examples.items()):
        seg, start = ex["turn"], ex["turn_start"]
        t = start + np.arange(len(seg)) / SR
        axes[row, 0].plot(t, seg, lw=0.4, color=COLORS[label])
        axes[row, 0].set_title(f"{label}: one caller turn, {len(seg) / SR:.1f} s = {len(seg):,} samples at 8,000 Hz", fontsize=9)
        axes[row, 0].set_xlabel("time (s)")
        axes[row, 0].set_ylabel("amplitude")
        # 25 ms of the loudest part
        n = int(0.025 * SR)
        energy = np.convolve(seg ** 2, np.ones(n), mode="valid")
        i = int(np.argmax(energy))
        z = seg[i:i + n]
        tz = (start + (i + np.arange(n)) / SR) * 1000
        axes[row, 1].plot(tz, z, color="#999", lw=1)
        axes[row, 1].plot(tz, z, "o", ms=3, color=COLORS[label])
        axes[row, 1].set_title(f"{label}: 25 ms zoom = {n} samples, one dot per sample (every 0.125 ms)", fontsize=9)
        axes[row, 1].set_xlabel("time (ms)")
    fig.suptitle("Waveforms: sound as a list of numbers (real calls from the training split)")
    fig.tight_layout()
    fig.savefig(FIG / "sp_waveform.png", dpi=130)
    plt.close(fig)


def fig_sampling_nyquist(examples):
    seg = examples["human"]["turn"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    f = 1000
    tc = np.linspace(0, 0.004, 4000)
    axes[0].plot(tc * 1000, np.sin(2 * np.pi * f * tc), color="#bbb", lw=2, label="true 1 kHz tone")
    t8 = np.arange(0, 0.004, 1 / 8000)
    axes[0].plot(t8 * 1000, np.sin(2 * np.pi * f * t8), "o", ms=4, color="#1f77b4", label="sampled at 8 kHz: 8 samples per period, fine")
    f2 = 5000
    axes[0].plot(tc * 1000, 0.6 * np.sin(2 * np.pi * f2 * tc), color="#f5b7b1", lw=1, label="a 5 kHz tone (above the 4 kHz Nyquist limit)")
    axes[0].plot(t8 * 1000, 0.6 * np.sin(2 * np.pi * f2 * t8), "s-", ms=4, color="#d62728",
                 label="the same 5 kHz tone sampled at 8 kHz looks like 3 kHz (aliasing)")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_title("Nyquist: 8,000 samples/s can only represent frequencies below 4,000 Hz", fontsize=9.5)
    axes[0].legend(fontsize=7, loc="lower left")
    # 8 kHz vs 16 kHz spectrum of the same speech
    up = audio.resample(seg, SR, 16000)
    for wave, sr, color, name in ((seg, 8000, "#1f77b4", "native 8 kHz (257 bins up to 4 kHz)"),
                                  (up, 16000, "#ff7f0e", "resampled to 16 kHz (513 bins up to 8 kHz)")):
        spec = np.mean(np.abs(librosa.stft(wave, n_fft=int(0.064 * sr), hop_length=int(0.01 * sr))) ** 2, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=int(0.064 * sr))
        axes[1].plot(freqs, 10 * np.log10(spec + 1e-12), color=color, lw=1, label=name)
    axes[1].axvline(4000, color="gray", ls="--")
    axes[1].text(4100, axes[1].get_ylim()[0] + 8, "nothing above 4 kHz:\nresampling adds no information", fontsize=7.5, color="gray")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("power (dB)")
    axes[1].set_title("The same caller turn at 8 kHz and after resampling to 16 kHz (what the backbones receive)", fontsize=9.5)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "sp_sampling_nyquist.png", dpi=130)
    plt.close(fig)


def fig_fft(examples):
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    t = np.arange(int(0.064 * SR)) / SR
    synth = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 900 * t) + 0.25 * np.sin(2 * np.pi * 2500 * t)
    cols = [(synth, "3 pure tones: 150 + 900 + 2500 Hz", "black")]
    for label, ex in examples.items():
        seg = ex["turn"]
        n = len(t)
        energy = np.convolve(seg ** 2, np.ones(n), mode="valid")
        i = int(np.argmax(energy))
        cols.append((seg[i:i + n], f"{label}: 64 ms of a loud vowel", COLORS[label]))
    freqs = scipy.fft.rfftfreq(len(t), 1 / SR)
    for c, (sig, name, color) in enumerate(cols):
        axes[0, c].plot(t * 1000, sig, lw=0.9, color=color)
        axes[0, c].set_title(f"time domain: {name}", fontsize=9)
        axes[0, c].set_xlabel("time (ms)")
        spectrum = np.abs(scipy.fft.rfft(sig * np.hanning(len(sig))))
        axes[1, c].plot(freqs, 20 * np.log10(spectrum + 1e-6), lw=0.9, color=color)
        axes[1, c].set_title("frequency domain: FFT magnitude", fontsize=9)
        axes[1, c].set_xlabel("frequency (Hz)")
        axes[1, c].set_ylabel("dB")
    axes[1, 1].text(0.35, 0.85, "evenly spaced peaks = pitch harmonics\nbroad bumps = formants", transform=axes[1, 1].transAxes, fontsize=7.5)
    fig.suptitle("The FFT re-describes a short piece of waveform as 'how much of each frequency'")
    fig.tight_layout()
    fig.savefig(FIG / "sp_fft.png", dpi=130)
    plt.close(fig)


def fig_representations(examples):
    for label, ex in examples.items():
        seg = ex["turn"]
        S_db, mel_db, mfcc = representations(seg)
        fig, axes = plt.subplots(4, 1, figsize=(10, 10.5))
        t = np.arange(len(seg)) / SR
        axes[0].plot(t, seg, lw=0.4, color=COLORS[label])
        axes[0].set_xlim(0, t[-1])
        axes[0].set_title(f"1) waveform: {len(seg):,} samples at 8,000 Hz = {len(seg) / SR:.2f} s")
        kw = dict(sr=SR, hop_length=HOP, n_fft=N_FFT, x_axis="time")
        im = librosa.display.specshow(S_db, y_axis="hz", ax=axes[1], cmap="magma", **kw)
        fig.colorbar(im, ax=axes[1], format="%+.0f dB")
        axes[1].set_title(f"2) spectrogram: {S_db.shape[0]} linear frequency bins (0-4 kHz) x {S_db.shape[1]} frames (10 ms)")
        im = librosa.display.specshow(mel_db, y_axis="mel", ax=axes[2], cmap="magma", **kw)
        fig.colorbar(im, ax=axes[2], format="%+.0f dB")
        axes[2].set_title(f"3) mel spectrogram: {N_MELS} mel bands x {mel_db.shape[1]} frames")
        im = librosa.display.specshow(mfcc, ax=axes[3], cmap="coolwarm", **kw)
        fig.colorbar(im, ax=axes[3])
        axes[3].set_ylabel("coefficient")
        axes[3].set_title(f"4) MFCC: {N_MFCC} coefficients x {mfcc.shape[1]} frames")
        fig.colorbar(plt.cm.ScalarMappable(), ax=axes[0]).remove()
        fig.suptitle(f"{label.upper()} caller turn [{ex['anon_id']}]: four views of the same 8 kHz audio", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / f"sp_representations_{label}.png", dpi=120)
        plt.close(fig)


def fig_spectrogram_pair(examples):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for c, (label, ex) in enumerate(examples.items()):
        S_db, mel_db, _ = representations(ex["turn"])
        kw = dict(sr=SR, hop_length=HOP, n_fft=N_FFT, x_axis="time")
        librosa.display.specshow(S_db, y_axis="hz", ax=axes[0, c], cmap="magma", **kw)
        axes[0, c].set_title(f"{label.upper()} - spectrogram (linear Hz)  [{ex['anon_id']}]", fontsize=9.5)
        librosa.display.specshow(mel_db, y_axis="mel", ax=axes[1, c], cmap="magma", **kw)
        axes[1, c].set_title(f"{label.upper()} - mel spectrogram", fontsize=9.5)
        axes[0, c].axhline(3400, color="cyan", ls=":", lw=0.8)
    fig.suptitle("Human vs synthetic caller turn side by side (dotted line: 3.4 kHz telephone band edge)")
    fig.tight_layout()
    fig.savefig(FIG / "sp_spectrogram_pair.png", dpi=130)
    plt.close(fig)


def fig_mel_mfcc_concepts(examples):
    seg = examples["human"]["turn"]
    _, mel_db, mfcc = representations(seg)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    hz = np.linspace(0, SR / 2, 400)
    axes[0].plot(hz, librosa.hz_to_mel(hz))
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("mel")
    axes[0].set_title("mel scale: equal steps sound equally far apart", fontsize=9.5)
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=20, fmax=SR / 2)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    for row in fb:
        axes[1].plot(freqs, row / row.max(), lw=1)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_title("20 of the mel filters: narrow low, wide high", fontsize=9.5)
    f = int(np.argmax(mel_db.mean(axis=0)))
    coeffs = scipy.fft.dct(mel_db[:, f], type=2, norm="ortho")
    smooth = scipy.fft.idct(np.r_[coeffs[:N_MFCC], np.zeros(N_MELS - N_MFCC)], type=2, norm="ortho")
    axes[2].plot(mel_db[:, f], label=f"log-mel spectrum of one frame ({N_MELS} values)")
    axes[2].plot(smooth, lw=2.5, label=f"rebuilt from {N_MFCC} MFCCs: the smooth envelope")
    axes[2].set_xlabel("mel band")
    axes[2].set_ylabel("dB")
    axes[2].set_title("MFCCs keep the spectral SHAPE, drop the fine detail", fontsize=9.5)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "sp_mel_mfcc_concepts.png", dpi=130)
    plt.close(fig)


def fig_pitch(n_per_class=15):
    """F0 statistics of caller speech on a subset of train calls (pyin is slow: ~seconds per call)."""
    stats = pd.read_csv(config.OUTPUTS_DIR / "dataset_inspection" / "call_stats.csv")
    rows = []
    for label in config.LABELS:
        d = stats[(stats["split"] == "train") & (stats["label"] == label)].sample(n=n_per_class, random_state=config.SEED)
        for anon_id in d["anon_id"]:
            caller, _, sr, turns = load_call(anon_id)
            regions = dataset.turns_of_channel(turns, 0)
            speech = np.concatenate([audio.slice_chunk(caller, sr, s, e) for s, e in regions])[:sr * 30]
            f0, voiced, _ = librosa.pyin(speech, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=160)
            f0 = f0[voiced & np.isfinite(f0)]
            if len(f0) > 20:
                rows.append({"anon_id": anon_id, "label": label, "f0_median": float(np.median(f0)),
                             "f0_std": float(np.std(f0)), "f0_iqr": float(np.subtract(*np.percentile(f0, [75, 25]))),
                             "voiced_fraction": float(voiced.mean()),
                             "f0_range_semitones": float(12 * np.log2(np.percentile(f0, 95) / np.percentile(f0, 5)))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pitch_stats.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (col, title) in zip(axes, [("f0_median", "median F0 (Hz)"), ("f0_std", "F0 standard deviation (Hz)"),
                                       ("f0_range_semitones", "F0 range, 5th-95th percentile (semitones)"), ("voiced_fraction", "voiced fraction")]):
        for label, color in COLORS.items():
            v = df.loc[df["label"] == label, col]
            ax.hist(v, bins=12, alpha=0.55, color=color, label=f"{label} (median {v.median():.3g})")
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7)
    fig.suptitle(f"Pitch statistics (pyin, first 30 s of caller speech, {n_per_class} train calls per class): illustration, not a classifier")
    fig.tight_layout()
    fig.savefig(FIG / "sp_pitch.png", dpi=130)
    plt.close(fig)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    picks = pick_examples()
    examples, info = {}, {}
    for label, row in picks.items():
        caller, agent, sr, turns = load_call(row["anon_id"])
        regions = audio.detect_speech(caller, sr, other=agent)
        turn, start = longest_turn(caller, sr, regions)
        examples[label] = {"anon_id": row["anon_id"], "caller": caller, "agent": agent, "sr": sr, "turns": turns,
                           "regions": regions, "turn": turn, "turn_start": start}
        sf.write(OUT / "examples" / f"{label}_{row['anon_id']}_turn.wav", turn, sr)
        info[label] = {"anon_id": row["anon_id"], "duration_s": len(caller) / sr, "caller_speech_s": float(row["caller_speech_s"]),
                       "turn_start_s": float(start), "turn_seconds": len(turn) / sr, "turn_samples": int(len(turn)),
                       "peak": float(np.abs(turn).max()), "rms_db": audio.rms_db(turn), "n_vad_regions": len(regions)}
        print(f"{label:<10} {row['anon_id']}  call {len(caller) / sr:.0f}s, caller speech {row['caller_speech_s']:.1f}s, "
              f"example turn at {start:.1f}s ({len(turn) / sr:.2f}s, {len(turn)} samples), rms {audio.rms_db(turn):.1f} dB")
    fig_call_overview(examples)
    fig_waveform_zoom(examples)
    fig_sampling_nyquist(examples)
    fig_fft(examples)
    fig_representations(examples)
    fig_spectrogram_pair(examples)
    fig_mel_mfcc_concepts(examples)
    print("pitch statistics (pyin on 30 calls, ~1-2 min) ...")
    pitch = fig_pitch()
    info["pitch_medians"] = pitch.groupby("label")[["f0_median", "f0_std", "f0_range_semitones", "voiced_fraction"]].median().to_dict()
    save_json(info, OUT / "examples.json")
    print(f"Saved figures sp_*.png in {FIG} and {OUT / 'examples.json'}")


if __name__ == "__main__":
    main()
