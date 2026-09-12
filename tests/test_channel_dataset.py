"""
Integrity tests for the generated channel dataset.

A mislabelled or leaked sample does not crash anything - it quietly produces a better-looking number than the
model deserves, and nobody finds out until production. These tests are the only thing standing between us and
that, so they check the boring things exhaustively:

  * no call id appears in two splits (a validation voice must never reach a gradient)
  * nothing from the UNSEEN codec family exists on the training side AT ALL
  * every transformed sample still carries the label and split of the call it came from
  * every sample maps back to a real original call, and no sample id is duplicated
  * the audio on disk really is 8 kHz mono PCM16 of the right duration
  * channel 1 of a reconstructed stereo file is the ORIGINAL agent, bit for bit - only channel 0 was
    transformed, and any claim to the contrary in a report would be false

They skip cleanly when the dataset has not been generated yet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config  # noqa: E402
import dataset as ds  # noqa: E402
from build_channel_dataset import MANIFESTS, PLAN, sample_name  # noqa: E402

ALL = MANIFESTS / "all.csv"
pytestmark = pytest.mark.skipif(not ALL.exists(), reason="channel dataset not generated yet")


@pytest.fixture(scope="module")
def man():
    df = pd.read_csv(ALL)
    return df[df["status"] == "ok"].reset_index(drop=True)


@pytest.fixture(scope="module")
def official():
    return ds.load_manifest().set_index("anon_id")


def test_dataset_is_not_empty(man):
    assert len(man) > 0


def test_no_call_appears_in_two_splits(man):
    """The single most important line in this file."""
    per_call = man.groupby("call_id")["split"].nunique()
    offenders = list(per_call[per_call > 1].index)
    assert not offenders, f"calls present in more than one split: {offenders[:10]}"


def test_split_matches_the_official_manifest(man, official):
    for r in man.itertuples():
        assert r.split == official.loc[r.call_id, "split"], \
            f"{r.sample_id} is in {r.split} but {r.call_id} is an official {official.loc[r.call_id, 'split']} call"


def test_label_is_preserved(man, official):
    for r in man.itertuples():
        assert r.label == official.loc[r.call_id, "label"], f"{r.sample_id} changed label"
        assert int(r.y) == config.LABEL2ID[r.label]


def test_no_unseen_family_on_the_training_side(man):
    """
    The generalisation claim is 'the model has never met these codecs'. The cheapest way to keep that true is
    for the files not to exist. PLAN encodes the intent; this checks the intent survived contact with the disk.
    """
    assert PLAN[("train", "unseen")] == 0
    leaked = man[(man["split"] == "train") & (man["family"] == "unseen")]
    assert leaked.empty, f"{len(leaked)} unseen-family samples exist in train: {list(leaked['sample_id'][:5])}"


def test_sample_ids_are_unique(man):
    dupes = man["sample_id"][man["sample_id"].duplicated()].tolist()
    assert not dupes, f"duplicate sample ids: {dupes[:10]}"


def test_sample_id_encodes_its_origin(man):
    for r in man.itertuples():
        assert r.sample_id == sample_name(r.call_id, r.family, r.variant)


def test_every_sample_maps_back_to_a_real_call(man, official):
    unknown = set(man["call_id"]) - set(official.index)
    assert not unknown, f"samples pointing at calls that do not exist: {sorted(unknown)[:5]}"


def test_class_balance_survived(man, official):
    """Transforming a dataset must not quietly drop one class."""
    for (split, family), g in man.groupby(["split", "family"]):
        assert g["label"].nunique() == 2, f"{split}/{family} only has {g['label'].unique()}"


def test_audio_files_exist_and_are_8k_mono(man):
    for r in man.sample(min(40, len(man)), random_state=0).itertuples():
        p = Path(r.channel_path)
        assert p.exists(), f"missing {p}"
        info = sf.info(str(p))
        assert info.samplerate == config.SOURCE_SAMPLE_RATE, f"{p.name} is {info.samplerate} Hz"
        assert info.channels == 1, f"{p.name} has {info.channels} channels"
        assert info.subtype == "PCM_16", f"{p.name} is {info.subtype}"


def test_duration_matches_the_original(man):
    for r in man.sample(min(40, len(man)), random_state=1).itertuples():
        orig = sf.info(str(r.original_path))
        chan = sf.info(str(r.channel_path))
        assert abs(chan.frames - orig.frames) <= 1, \
            f"{r.sample_id}: {chan.frames} samples vs {orig.frames} in the original"


def test_channel_audio_is_not_identical_to_the_original(man):
    """If a sample came out bit-identical, the channel silently did nothing to it."""
    for r in man.sample(min(15, len(man)), random_state=2).itertuples():
        orig, _ = sf.read(str(r.original_path), dtype="float32", always_2d=True)
        chan, _ = sf.read(str(r.channel_path), dtype="float32", always_2d=True)
        n = min(len(orig), len(chan))
        assert not np.allclose(orig[:n, 0], chan[:n, 0], atol=1e-4), f"{r.sample_id} was not transformed"


def test_reconstructed_stereo_keeps_the_original_agent_channel(man):
    """
    Channel 1 of a reconstructed file is the agent EXACTLY as the challenge recorded it - it did not go
    through a codec. The report says so, so it had better be true.
    """
    rows = man[man["stereo_path"].astype(str).str.len() > 0]
    if rows.empty:
        pytest.skip("no reconstructed stereo files")
    for r in rows.sample(min(15, len(rows)), random_state=3).itertuples():
        p = Path(r.stereo_path)
        assert p.exists(), f"missing {p}"
        rebuilt, sr = sf.read(str(p), dtype="int16", always_2d=True)
        assert sr == config.SOURCE_SAMPLE_RATE and rebuilt.shape[1] == 2
        original, _ = sf.read(str(r.original_path), dtype="int16", always_2d=True)
        n = min(len(rebuilt), len(original))
        assert np.array_equal(rebuilt[:n, 1], original[:n, config.AGENT_CHANNEL]), \
            f"{r.sample_id}: the agent channel was modified"
        assert not np.array_equal(rebuilt[:n, 0], original[:n, config.CALLER_CHANNEL]), \
            f"{r.sample_id}: the caller channel was NOT transformed"


def test_no_residual_delay_after_alignment(man):
    """
    The real test of the rebuild: after apply_channel has shifted the transformed caller back, is there any
    timing error LEFT? This and not the correlation is what decides whether channel 0 can sit next to the
    untouched agent channel - an AGC legitimately flattens the envelope and drags the correlation down to
    ~0.4 while leaving the timing perfect to within a sample or two.
    """
    residual = man["residual_delay_samples"].abs()
    assert residual.max() <= 8, \
        f"{int((residual > 8).sum())} samples still misaligned after the shift, worst {residual.max()} samples"


def test_measured_delay_is_physically_plausible(man):
    """Codec algorithmic delay is tens of milliseconds. Hundreds would mean the aligner is chasing an echo."""
    assert man["estimated_delay_ms"].abs().quantile(0.99) < 200
    assert man["estimated_delay_ms"].abs().max() < 260, "the search window itself is +-250 ms"


def test_every_planned_sample_was_generated(man):
    official = ds.load_manifest()
    for (split, family), n_variants in PLAN.items():
        expected = len(official[official["split"] == split]) * n_variants
        got = len(man[(man["split"] == split) & (man["family"] == family)])
        assert got == expected, f"{split}/{family}: expected {expected} samples, found {got}"
