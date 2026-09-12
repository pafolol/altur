import numpy as np
import pytest
from behavior.config import VADConfig
from behavior.vad import probabilities_to_turns, speech_iou


def test_frame_segmentation_hysteresis_and_empty_channel():
    p = np.zeros((2, 100))
    p[0, 10:30] = .9
    p[0, 20:22] = .1  # 64ms internal silence should be bridged
    config = VADConfig(pad_s=0, merge_gap_s=0)
    turns = probabilities_to_turns(p, 3.2, config)
    assert len(turns) == 1
    assert turns[0].start == .32 and turns[0].end == .96
    assert turns[0].channel == 0


def test_short_speech_dropped_and_end_clipped():
    p = np.zeros((2, 10))
    p[0, 2] = 1
    p[1, :] = 1
    turns = probabilities_to_turns(p, .3)
    assert len(turns) == 1 and turns[0].end == .3 and turns[0].channel == 1
    assert speech_iou(turns, turns, 1) == 1


def test_wrong_vad_model_rejected_offline(tmp_path):
    from behavior.vad import SileroVAD
    path = tmp_path / "wrong.onnx"
    path.write_bytes(b"not the pinned model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        SileroVAD(path)
