"""Load Metrica Sports sample tracking and event data (Sample Games 1 and 2).

Metrica coordinates are normalised to [0, 1]; we convert to metres on a
105 x 68 m pitch. Frame rate is 25 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PITCH_L, PITCH_W, FPS = 105.0, 68.0, 25.0


@dataclass
class Game:
    name: str
    home: np.ndarray    # (T, n_players, 2) metres, NaN when player not on pitch
    away: np.ndarray    # (T, n_players, 2)
    ball: np.ndarray    # (T, 2)
    period: np.ndarray  # (T,)
    frame: np.ndarray   # (T,) frame numbers as in the source files
    events: pd.DataFrame


def _read_tracking(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw = pd.read_csv(path, header=None, skiprows=3)
    meta = raw.iloc[:, :3].copy()
    meta.columns = ["period", "frame", "time"]
    xy = raw.iloc[:, 3:].to_numpy(dtype=float)
    n = xy.shape[1] // 2
    xy = xy.reshape(len(raw), n, 2)
    xy[..., 0] *= PITCH_L
    xy[..., 1] *= PITCH_W
    return meta, xy[:, :-1, :], xy[:, -1, :]  # players, ball


def load_game(data_dir: str | Path, game: int) -> Game:
    """Load Sample_Game_{game} (1 or 2) from the Metrica `data/` folder."""
    d = Path(data_dir) / f"Sample_Game_{game}"
    p = f"Sample_Game_{game}"
    meta, home, ball = _read_tracking(d / f"{p}_RawTrackingData_Home_Team.csv")
    _, away, _ = _read_tracking(d / f"{p}_RawTrackingData_Away_Team.csv")
    events = pd.read_csv(d / f"{p}_RawEventsData.csv")
    return Game(p, home, away, ball, meta["period"].to_numpy(),
                meta["frame"].to_numpy(), events)
