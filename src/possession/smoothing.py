"""Temporal smoothing: possession is sticky, so frame-by-frame flicker is noise."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_median(pred: np.ndarray, window: int = 25) -> np.ndarray:
    """Median filter over ``window`` frames (25 frames = 1 s at 25 Hz)."""
    s = pd.Series(pred)
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()


def viterbi_two_state(p_home: np.ndarray, stay: float = 0.995) -> np.ndarray:
    """Most likely home/away sequence for per-frame probabilities ``p_home``.

    A two-state HMM whose emissions are the frame-level probabilities and whose
    transition matrix strongly prefers staying in the same state. NaNs are
    treated as uninformative (0.5).
    """
    p = np.clip(np.nan_to_num(p_home, nan=0.5), 1e-4, 1 - 1e-4)
    log_e = np.stack([np.log(1 - p), np.log(p)], axis=1)  # states: 0 away, 1 home
    log_stay, log_move = np.log(stay), np.log(1 - stay)
    T = len(p)
    score = np.zeros((T, 2))
    back = np.zeros((T, 2), dtype=int)
    score[0] = log_e[0]
    for t in range(1, T):
        for s in (0, 1):
            stay_sc = score[t - 1, s] + log_stay
            move_sc = score[t - 1, 1 - s] + log_move
            back[t, s] = s if stay_sc >= move_sc else 1 - s
            score[t, s] = max(stay_sc, move_sc) + log_e[t, s]
    path = np.zeros(T, dtype=int)
    path[-1] = score[-1].argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path.astype(float)


def distance_to_prob(d_home: np.ndarray, d_away: np.ndarray, k: float = 0.8) -> np.ndarray:
    """Turn the distance gap into P(home) with a logistic curve."""
    return 1.0 / (1.0 + np.exp(-k * (d_away - d_home)))
