"""Proxy ground truth: derive per-frame possession from the event data.

The sample data has no explicit possession label, so we build one:

* PASS / SHOT / RECOVERY / SET PIECE start possession for that event's team,
  and it is carried forward frame by frame until the next event.
* BALL LOST / BALL OUT / CHALLENGE / FAULT RECEIVED make the label
  *undetermined* (NaN) until the next possession-starting event.

Undetermined frames are excluded from evaluation. This is an approximation:
a model scored against these labels is only as good as the labels themselves.
"""
from __future__ import annotations

import numpy as np

from .data import Game

STARTS = {"PASS", "SHOT", "RECOVERY", "SET PIECE"}
BREAKS = {"BALL LOST", "BALL OUT", "CHALLENGE", "FAULT RECEIVED"}


def possession_labels(game: Game) -> np.ndarray:
    """Return array (T,) with 1 = home, 0 = away, NaN = undetermined."""
    ev = game.events.sort_values("Start Frame", kind="stable")
    frames = game.frame
    labels = np.full(len(frames), np.nan)
    current = np.nan
    ev_frames = ev["Start Frame"].to_numpy()
    ev_types = ev["Type"].to_numpy()
    ev_teams = ev["Team"].to_numpy()
    j = 0
    for i, f in enumerate(frames):
        while j < len(ev) and ev_frames[j] <= f:
            if ev_types[j] in STARTS:
                current = 1.0 if ev_teams[j] == "Home" else 0.0
            elif ev_types[j] in BREAKS:
                current = np.nan
            j += 1
        labels[i] = current
    return labels
