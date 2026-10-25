"""Run the possession experiments on Metrica Sample Games 1 and 2.

Usage:
    python run_experiments.py --data-dir path/to/sample-data/data

Evaluates on frames where a proxy label exists (see src/possession/labels.py).
The learned model is trained on one game and tested on the other (match-level
split), never on random frames of the same match.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from possession.baseline import nearest_player, nearest_with_threshold, team_distances  # noqa: E402
from possession.data import PITCH_L, PITCH_W, load_game  # noqa: E402
from possession.features import build_features  # noqa: E402
from possession.labels import possession_labels  # noqa: E402
from possession.smoothing import distance_to_prob, rolling_median, viterbi_two_state  # noqa: E402


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    m = ~np.isnan(pred) & ~np.isnan(truth)
    return float((pred[m] == truth[m]).mean())


def flips(pred: np.ndarray) -> int:
    p = pred[~np.isnan(pred)]
    return int((np.diff(p) != 0).sum())


def draw_pitch(ax):
    ax.set_xlim(0, PITCH_L); ax.set_ylim(PITCH_W, 0); ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), PITCH_L, PITCH_W, fill=False, ec="k"))
    ax.axvline(PITCH_L / 2, c="k", lw=.8)
    ax.add_patch(plt.Circle((PITCH_L / 2, PITCH_W / 2), 9.15, fill=False, ec="k", lw=.8))
    ax.axis("off")


def plot_frame(game, frame_idx, pred, out):
    fig, ax = plt.subplots(figsize=(8, 5.2)); draw_pitch(ax)
    ax.scatter(*game.home[frame_idx].T, c="tab:red", s=60, label="Home")
    ax.scatter(*game.away[frame_idx].T, c="tab:blue", s=60, label="Away")
    ax.scatter(*game.ball[frame_idx], c="k", s=40, marker="o", label="Ball")
    team = {1.0: "Home", 0.0: "Away"}.get(pred[frame_idx], "?")
    ax.set_title(f"{game.name}, frame {game.frame[frame_idx]}: predicted possession = {team}")
    ax.legend(loc="lower center", ncol=3, frameon=False)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_timeline(truth, preds: dict, out, fps=25):
    fig, axes = plt.subplots(len(preds) + 1, 1, figsize=(11, 1.0 * (len(preds) + 1)), sharex=True)
    t = np.arange(len(truth)) / fps / 60
    for ax, (name, y) in zip(axes, [("Proxy truth", truth), *preds.items()]):
        ax.fill_between(t, 0, 1, where=y == 1, color="tab:red", step="mid", lw=0)
        ax.fill_between(t, 0, 1, where=y == 0, color="tab:blue", step="mid", lw=0)
        ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("minutes (red = home, blue = away, white = undetermined)")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    games = {g: load_game(a.data_dir, g) for g in (1, 2)}
    truth = {g: possession_labels(games[g]) for g in games}
    dist = {g: team_distances(games[g]) for g in games}
    feats = {g: build_features(games[g]) for g in games}

    rows = []  # (method, game, accuracy, flips)
    preds_by_game = {}
    for g, game in games.items():
        dh, da = dist[g]
        base = nearest_player(dh, da)
        thr = nearest_with_threshold(dh, da, 3.0)
        med = rolling_median(thr, 75)
        vit = viterbi_two_state(distance_to_prob(dh, da))
        vit[np.isnan(dh) | np.isnan(da)] = np.nan
        methods = {"Nearest player": base, "+ 3 m threshold, hold last": thr,
                   "+ rolling median (3 s)": med, "Viterbi HMM on distance gap": vit}
        # learned model: train on the other game, test on this one
        import lightgbm as lgb
        other = 2 if g == 1 else 1
        tr_mask = ~np.isnan(truth[other]) & feats[other].notna().all(axis=1).to_numpy()
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                                 verbose=-1, random_state=0)
        clf.fit(feats[other][tr_mask], truth[other][tr_mask])
        p = np.full(len(game.frame), np.nan)
        ok = feats[g].notna().all(axis=1).to_numpy()
        p[ok] = clf.predict_proba(feats[g][ok])[:, 1]
        lgb_raw = np.where(np.isnan(p), np.nan, (p > .5).astype(float))
        lgb_vit = viterbi_two_state(p, stay=0.99)
        lgb_vit[np.isnan(p)] = np.nan
        methods["LightGBM (trained on the other game)"] = lgb_raw
        methods["LightGBM + Viterbi"] = lgb_vit
        for name, y in methods.items():
            rows.append((name, g, accuracy(y, truth[g]), flips(y)))
        preds_by_game[g] = methods

    # results table
    names = list(dict.fromkeys(r[0] for r in rows))
    lines = ["| Method | Game 1 acc | Game 2 acc | Game 1 flips | Game 2 flips |", "|---|---|---|---|---|"]
    for n in names:
        r = {g: (acc, fl) for (m, g, acc, fl) in rows if m == n}
        lines.append(f"| {n} | {r[1][0]:.3f} | {r[2][0]:.3f} | {r[1][1]} | {r[2][1]} |")
    n_lab = {g: int((~np.isnan(truth[g])).sum()) for g in games}
    lines.append(f"\nLabelled frames: game 1 = {n_lab[1]:,}, game 2 = {n_lab[2]:,}. "
                 f"'Flips' = number of possession changes predicted (fewer = less flicker).")
    table = "\n".join(lines)
    (out / "results.md").write_text(table + "\n")
    print(table)

    # figures (game 1)
    g = 1; game = games[g]
    plot_frame(game, 4000, preds_by_game[g]["+ rolling median (3 s)"], out / "frame_example.png")
    sel = ["Nearest player", "+ rolling median (3 s)", "Viterbi HMM on distance gap",
           "LightGBM + Viterbi"]
    s = slice(0, 25 * 60 * 10)  # first 10 minutes
    plot_timeline(truth[g][s], {k: preds_by_game[g][k][s] for k in sel}, out / "timeline_game1.png")


if __name__ == "__main__":
    main()
