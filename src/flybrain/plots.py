"""Figures for the results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_closed_loop(log: dict, food_xy, path: str | Path) -> Path:
    """Trajectory + odor + brain readout + activity of a closed-loop run."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    sc = a.scatter(log["x"], log["y"], c=log["t"], s=8, cmap="viridis")
    a.plot(*food_xy, marker="*", ms=18, color="tab:orange", label="food (odor source)")
    a.plot(log["x"][0], log["y"][0], "ko", label="start")
    a.set_aspect("equal")
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    a.set_title("Trajectory (colour = time, s)")
    a.legend(loc="best", fontsize=8)
    fig.colorbar(sc, ax=a)

    a = ax[0, 1]
    a.plot(log["t"], log["odor_l"], label="left sensors")
    a.plot(log["t"], log["odor_r"], label="right sensors")
    a.set_xlabel("time (s)")
    a.set_ylabel("odor intensity")
    a.set_title("Sensory input")
    a.legend(fontsize=8)

    a = ax[1, 0]
    contrast = (log["odor_l"] - log["odor_r"]) / (log["odor_l"] + log["odor_r"] + 1e-9)
    a.plot(log["t"], log["turn"], label="DN steering readout (+ = left)")
    a2 = a.twinx()
    a2.plot(log["t"], contrast, color="tab:red", alpha=0.6, label="odor L/R contrast")
    a.set_xlabel("time (s)")
    a.set_title("Brain -> body: steering command")
    a.legend(loc="upper left", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)

    a = ax[1, 1]
    a.plot(log["t"], log["n_spikes"], label="all spikes / 25 ms")
    a.plot(log["t"], log["dn_spikes"], label="descending-neuron spikes / 25 ms")
    a.set_yscale("log")
    a.set_xlabel("time (s)")
    a.set_title("Whole-brain activity")
    a.legend(fontsize=8)

    if len(contrast) > 3 and np.std(contrast) > 0 and np.std(log["turn"]) > 0:
        r = np.corrcoef(contrast, log["turn"])[0, 1]
        fig.suptitle(f"Closed loop: corr(odor contrast, brain steering) = {r:+.2f}")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
