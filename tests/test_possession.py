import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from possession.baseline import nearest_player, nearest_with_threshold  # noqa: E402
from possession.smoothing import rolling_median, viterbi_two_state  # noqa: E402


def test_nearest_player_picks_closer_team():
    d_home = np.array([1.0, 5.0, np.nan])
    d_away = np.array([4.0, 2.0, 3.0])
    out = nearest_player(d_home, d_away)
    assert out[0] == 1.0 and out[1] == 0.0 and np.isnan(out[2])


def test_threshold_holds_last_possession():
    d_home = np.array([1.0, 8.0, 8.0])
    d_away = np.array([9.0, 9.0, 9.0])
    out = nearest_with_threshold(d_home, d_away, max_dist=3.0)
    assert list(out) == [1.0, 1.0, 1.0]  # ball far from everyone: keep home


def test_rolling_median_removes_single_frame_flicker():
    x = np.array([1, 1, 1, 0, 1, 1, 1], dtype=float)
    assert rolling_median(x, window=3)[3] == 1.0


def test_viterbi_ignores_isolated_noise_but_follows_real_change():
    p = np.array([0.9] * 20 + [0.1] + [0.9] * 20 + [0.1] * 40)
    path = viterbi_two_state(p, stay=0.99)
    assert path[20] == 1.0      # single noisy frame ignored
    assert path[-1] == 0.0      # sustained change followed
