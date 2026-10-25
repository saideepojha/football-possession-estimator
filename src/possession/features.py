"""Frame-level features for a learned possession model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import team_distances
from .data import FPS, Game


def _k_nearest(players: np.ndarray, ball: np.ndarray, k: int = 2) -> np.ndarray:
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), 99.0, d)
    return np.sort(d, axis=1)[:, :k]


def build_features(game: Game) -> pd.DataFrame:
    """Distances, ball kinematics, local player counts and short rolling context."""
    d_home, d_away = team_distances(game)
    kh = _k_nearest(game.home, game.ball)
    ka = _k_nearest(game.away, game.ball)
    ball = pd.DataFrame(game.ball, columns=["bx", "by"]).interpolate(limit=10)
    speed = np.hypot(ball["bx"].diff(), ball["by"].diff()) * FPS

    f = pd.DataFrame({
        "d_home": d_home, "d_away": d_away,
        "d_home2": kh[:, 1], "d_away2": ka[:, 1],
        "d_gap": d_away - d_home,
        "ball_x": ball["bx"], "ball_y": ball["by"],
        "ball_speed": speed,
        "n_home_5m": (np.linalg.norm(game.home - game.ball[:, None, :], axis=2) < 5).sum(1),
        "n_away_5m": (np.linalg.norm(game.away - game.ball[:, None, :], axis=2) < 5).sum(1),
    })
    # closing velocity of the nearest player of each team (m/s, negative = approaching)
    f["home_closing"] = f["d_home"].diff() * FPS
    f["away_closing"] = f["d_away"].diff() * FPS
    # short-term context: possession is persistent
    for w in (25, 75):
        f[f"gap_mean_{w}"] = f["d_gap"].rolling(w, center=True, min_periods=1).mean()
    return f
