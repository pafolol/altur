import pytest
from behavior.turns import Turn, normalize_turns, speaking_states, intersection_duration


def test_union_clip_and_channel_separation():
    turns = [Turn(1, 2, 0), Turn(1.5, 3, 0), Turn(3.1, 4, 0), Turn(1.5, 2.5, 1)]
    normalized = normalize_turns(turns, 3.5, .15)
    assert normalized == [Turn(1, 3.5, 0), Turn(1.5, 2.5, 1)]
    states = speaking_states(normalized, 3.5)
    assert states == [(0., 1, 0), (1, 1.5, 1), (1.5, 2.5, 3), (2.5, 3.5, 1)]
    assert intersection_duration(normalized[:1], normalized[1:]) == 1


@pytest.mark.parametrize("args", [(1, 1, 0), (-1, 2, 0), (0, 1, 2), (0, float('nan'), 0)])
def test_invalid_intervals(args):
    with pytest.raises(ValueError):
        Turn(*args)
