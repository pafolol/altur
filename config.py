"""
Central configuration of the Altur acoustic detector.

Every script under src/ reads its settings from here. Command-line flags of the scripts can override
a few of them for smoke tests (run any script with --help).

The dataset never moves: it stays in D:/altur/hackmty26 (manifest + turns) and
D:/altur/altur-challenge-audio/audio (the unzipped WAVs).
"""
import os
import random
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
CHALLENGE_DIR = PROJECT_ROOT.parent / "hackmty26"                       # official repo: manifest.csv + turns/
AUDIO_DIR = PROJECT_ROOT.parent / "altur-challenge-audio" / "audio"    # unzipped altur-challenge-audio.zip
MANIFEST = CHALLENGE_DIR / "manifest.csv"
TURNS_DIR = CHALLENGE_DIR / "turns"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENT_LOG = REPORTS_DIR / "EXPERIMENT_LOG.md"


# ----------------------------------------------------------------------------- .env
def load_env(path=PROJECT_ROOT / ".env"):
    """
    Read KEY=VALUE lines from the repository's .env, if there is one.

    No dependency, on purpose - this is the same few lines the semantic module uses, and adding
    python-dotenv to run a demo is not a trade worth making. `setdefault` means a real environment
    variable always wins over the file, so `set SEMANTIC_URL=... && python src/server.py` still
    overrides whatever the file says.

    The file holds credentials (Twilio, and any service URL you do not want to retype) and is
    git-ignored. `.env.example` lists every variable that does anything.
    """
    if not path.exists():
        return {}
    loaded = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
            loaded[key] = value
    return loaded


load_env()

# Hugging Face cache inside the project (keeps the models next to the code and off the C: drive).
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

# ----------------------------------------------------------------------------- data facts (verified by inspect_dataset.py)
SEED = 42
LABELS = ["human", "synthetic"]           # class id 1 = SYNTHETIC = the positive class we detect
LABEL2ID = {"human": 0, "synthetic": 1}
SPLITS = ["train", "val"]
SOURCE_SAMPLE_RATE = 8_000                # telephone audio, 8 kHz -> nothing above 4 kHz exists
MODEL_SAMPLE_RATE = 16_000                # every backbone was pretrained on 16 kHz waveforms
CALLER_CHANNEL = 0                        # channel 0 = caller (what we classify), channel 1 = agent
AGENT_CHANNEL = 1

# ----------------------------------------------------------------------------- segmentation (identical for every model)
# Voice-activity detection on the caller channel. No metadata is needed, so the same code runs at inference.
#   "silero" = Silero VAD (small neural network, robust to noisy caller channels) with the energy VAD as fallback
#   "energy" = frame-energy detector below (fails on calls with a high stationary noise floor: see inspect_dataset)
VAD_METHOD = "energy"
SILERO_THRESHOLD = 0.5
SILERO_MIN_SILENCE_MS = 300               # pauses shorter than this stay inside one speech region
SILERO_MIN_SPEECH_MS = 250
SILERO_PAD_MS = 100
VAD_FRAME_S = 0.02                        # analysis frame: 20 ms
VAD_HOP_S = 0.01                          # a new frame every 10 ms
VAD_THRESHOLD_DB = 15.0                   # a frame is speech if it is this many dB above the noise floor ... (12 dB failed on 21 noisy calls, 15 dB on 1: see inspect_dataset)
VAD_MIN_ABS_DB = -55.0                    # ... and above this absolute level (guards near-silent channels)
VAD_MIN_SPEECH_S = 0.30                   # speech regions shorter than this are dropped
VAD_MIN_GAP_S = 0.30                      # gaps shorter than this merge neighbouring regions into one turn
VAD_PAD_S = 0.10                          # padding added before/after every region
# Crosstalk gate: the agent's voice leaks (echo) into the caller channel on some lines, loud enough to look like
# caller speech. A caller frame only counts when caller_db >= agent_db - margin. None = gate off.
VAD_CROSSTALK_MARGIN_DB = None            # tested (0/3/6 dB): removes human speech during overlaps, does not fix noisy calls -> off

# Each caller speech region is cut into chunks the backbone sees one at a time.
CHUNK_SECONDS = 4.0                       # chunk length (same regime as the previous project)
MIN_CHUNK_SECONDS = 1.0                   # leftovers shorter than this are dropped
MAX_CHUNKS_PER_CALL = 60                  # safety cap for very long calls (never reached in this dataset)
# Every chunk is scaled to the same RMS level before any backbone sees it. Reason: synthetic callers in this
# dataset are ~6 dB louder than humans (inspect_dataset.py) - a trivial shortcut - and only two of the three
# backbones normalise their input themselves; without this step the comparison would not be fair.
TARGET_RMS_DB = -26.0

# ----------------------------------------------------------------------------- backbones (the experimental variable)
BACKBONES = {
    "wavlm": {
        "hf_name": "microsoft/wavlm-base-plus",
        "display": "WavLM Base+",
        "pretraining": "94k h English (Libri-Light, GigaSpeech, VoxPopuli-en), masked prediction + denoising",
    },
    "wav2vec2_spanish": {
        "hf_name": "facebook/wav2vec2-base-es-voxpopuli-v2",
        "display": "Wav2Vec2 Spanish (VoxPopuli)",
        "pretraining": "21.4k h Spanish European-Parliament speech (VoxPopuli), contrastive masked prediction",
    },
    "xlsr": {
        "hf_name": "facebook/wav2vec2-xls-r-300m",
        "display": "XLS-R 300M",
        "pretraining": "436k h in 128 languages (VoxPopuli, MLS, CommonVoice, BABEL, VoxLingua107)",
    },
}
BACKBONE_ORDER = ["wavlm", "wav2vec2_spanish", "xlsr"]

# ----------------------------------------------------------------------------- downstream classifiers (identical for every model)
LOGREG_C = 1.0                            # linear probe: inverse regularisation strength
MLP_HIDDEN = 256
MLP_DROPOUT = 0.3
MLP_EPOCHS = 60
MLP_BATCH_SIZE = 64
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_PATIENCE = 15                         # early stopping on validation loss
CHUNK_AGGREGATION = "mean"                # default call-level aggregation of chunk log-odds

# ----------------------------------------------------------------------------- endpoint contract
# The challenge returns {"is_synthetic": bool, "confidence": float}.  "confidence" is not defined more
# precisely in the PDF, so it is configurable:
#   "verdict"  -> probability that the returned verdict is right = max(p, 1 - p)   (default)
#   "synthetic"-> probability that the caller is synthetic (= the raw score)
CONFIDENCE_MODE = "verdict"
DECISION_THRESHOLD = 0.5                  # on the calibrated synthetic probability

# ----------------------------------------------------------------------------- how EARLY, not just what
# DECISION_THRESHOLD answers "which side of the fence". These two answer "is it far enough from the fence
# to act on", which is a different question and the one the latency story needs: a call that is 0.51
# synthetic after one chunk has not been detected, it has been guessed at.
#
# THESE ARE A POLICY CHOICE, NOT A FITTED VALUE, and saying so matters. They are a symmetric band around
# the decision threshold, wide enough that crossing it is a commitment rather than noise. They are NOT
# tuned on the validation split - tuning a "confidence" band on the same 71 calls used to report accuracy
# would make the resulting latency numbers meaningless. src/latency.py reports, for whatever band is set
# here, how often the first crossing turned out to agree with the final verdict; that is the honest check,
# and it is measured rather than assumed. Raise the band and detection is later but surer; lower it and
# the reverse. Both are live-tunable per request on /demo.
CONFIDENT_SYNTHETIC_THRESHOLD = 0.85      # at or above this, "synthetic" is a commitment
CONFIDENT_HUMAN_THRESHOLD = 0.15          # at or below this, "human" is a commitment

# ----------------------------------------------------------------------------- fine-tuning (optional stage, best frozen model only)
FT_BATCH_SIZE = 8
FT_HEAD_EPOCHS = 3
FT_UNFROZEN_EPOCHS = 8
FT_UNFREEZE_LAST_N_LAYERS = 4
FT_HEAD_LR = 1e-3
FT_BACKBONE_LR = 1e-5
FT_WEIGHT_DECAY = 0.01
FT_WARMUP_FRACTION = 0.1
FT_PATIENCE = 3
FT_MAX_GRAD_NORM = 1.0


# ----------------------------------------------------------------------------- helpers
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(verbose=True):
    import torch
    if torch.cuda.is_available():
        if verbose:
            print(f"[device] CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if verbose:
        print("[device] no CUDA GPU -> CPU")
    return torch.device("cpu")


def ensure_dirs():
    for d in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, EMBEDDINGS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
