"""Rule-based possession estimators."""
from __future__ import annotations

import numpy as np

from .data import Game


def team_distances(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the ball to the nearest home / away player, per frame (m)."""
    def nearest(players: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(players - game.ball[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        m = d.min(axis=1)
        m[~np.isfinite(game.ball).all(axis=1)] = np.nan  # ball not tracked
        return m
    return nearest(game.home), nearest(game.away)


def nearest_player(d_home: np.ndarray, d_away: np.ndarray) -> np.ndarray:
    """Possession = team of the nearest player (1 home, 0 away, NaN no ball)."""
    out = (d_home < d_away).astype(float)
    out[np.isnan(d_home) | np.isnan(d_away)] = np.nan
    return out


def nearest_with_threshold(d_home: np.ndarray, d_away: np.ndarray,
                           max_dist: float = 3.0) -> np.ndarray:
    """Nearest-player rule that only *changes* possession when a player is
    within ``max_dist`` metres of the ball; otherwise keep the previous team."""
    raw = nearest_player(d_home, d_away)
    d_min = np.fmin(d_home, d_away)
    out = np.full_like(raw, np.nan)
    last = np.nan
    for i in range(len(raw)):
        if not np.isnan(raw[i]) and d_min[i] <= max_dist:
            last = raw[i]
        out[i] = last
    return out
