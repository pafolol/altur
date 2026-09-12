"""
Dataset layer for the Altur HackMTY 2026 challenge.

  manifest.csv      one row per call: anon_id, label (human|synthetic), split (train|val), duration_s
  audio/<id>.wav    stereo 8 kHz 16-bit PCM; channel 0 = caller (classified), channel 1 = agent
  turns/<id>.json   {"turns": [{"channel": 0, "start": 12.4, "end": 15.1}, ...]}  (auto-derived, a starting point)

The official split is speaker-disjoint (README) and is used unchanged. There are NO speaker ids, TTS
system ids or codec fields, so every leakage check has to be done on the audio itself (inspect_dataset.py).
"""
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import config

UNREADABLE_LIST = config.OUTPUTS_DIR / "dataset_inspection" / "unreadable_files.txt"


def load_manifest():
    df = pd.read_csv(config.MANIFEST)
    df["y"] = df["label"].map(config.LABEL2ID).astype(int)
    df["path"] = [str(config.AUDIO_DIR / f"{i}.wav") for i in df["anon_id"]]
    df["turns_path"] = [str(config.TURNS_DIR / f"{i}.json") for i in df["anon_id"]]
    if UNREADABLE_LIST.exists():
        bad = set(UNREADABLE_LIST.read_text().split())
        df = df[~df["anon_id"].isin(bad)]
    return df.sort_values("anon_id").reset_index(drop=True)


def load_split(split, fraction=1.0, seed=config.SEED):
    """One official split; `fraction` < 1 gives a stratified (per label) seeded subset for smoke tests."""
    df = load_manifest()
    df = df[df["split"] == split]
    if fraction < 1.0:  # pandas 3 drops the grouping column inside groupby.apply -> sample per class by hand
        parts = [g.sample(n=max(2, int(round(len(g) * fraction))), random_state=seed) for _, g in df.groupby("label")]
        df = pd.concat(parts)
    return df.sort_values("anon_id").reset_index(drop=True)


def load_turns(turns_path):
    return json.loads(open(turns_path).read())["turns"]


def turns_of_channel(turns, channel):
    return sorted((t["start"], t["end"]) for t in turns if t["channel"] == channel)


def turn_stats(turns, total_s):
    """Conversation-structure numbers from the official turn file (used for inspection + shortcut check)."""
    caller = turns_of_channel(turns, config.CALLER_CHANNEL)
    agent = turns_of_channel(turns, config.AGENT_CHANNEL)
    caller_s = sum(e - s for s, e in caller)
    agent_s = sum(e - s for s, e in agent)
    # response latency: gap between the end of an agent turn and the start of the next caller turn
    latencies = []
    for cs, ce in caller:
        prev = [ae for _, ae in agent if ae <= cs]
        if prev:
            latencies.append(cs - max(prev))
    # overlap: seconds where caller and agent speak at the same time
    hop = 0.01
    from audio import regions_to_mask
    overlap_s = float(np.sum(regions_to_mask(caller, total_s, hop) & regions_to_mask(agent, total_s, hop)) * hop)
    return {
        "n_caller_turns": len(caller),
        "n_agent_turns": len(agent),
        "caller_speech_s": caller_s,
        "agent_speech_s": agent_s,
        "caller_speech_fraction": caller_s / total_s if total_s else 0.0,
        "mean_caller_turn_s": caller_s / len(caller) if caller else 0.0,
        "max_caller_turn_s": max((e - s for s, e in caller), default=0.0),
        "first_caller_start_s": caller[0][0] if caller else total_s,
        "last_caller_end_s": caller[-1][1] if caller else 0.0,
        "median_response_latency_s": float(np.median(latencies)) if latencies else np.nan,
        "std_response_latency_s": float(np.std(latencies)) if len(latencies) > 1 else np.nan,
        "overlap_s": overlap_s,
    }
