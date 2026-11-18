"""In-silico experiments on the whole-brain LIF model.

Each function returns a tidy ``pandas.DataFrame`` so results can be saved, plotted
and quoted. They are exposed through the ``flybrain`` command line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Connectome
from .lif import LIFNetwork


def _side_groups(c: Connectome, column: str, value: str):
    idx = c.select(column, value)
    side = c.neurons["side"].to_numpy()
    return idx[side[idx] == "left"], idx[side[idx] == "right"]


def descending_sides(c: Connectome):
    """Left / right descending neurons (the brain's motor output channels)."""
    return _side_groups(c, "super_class", "descending")


def sensory_activation(
    c: Connectome,
    stimulate,
    rate_hz: float = 150.0,
    duration_ms: float = 1000.0,
    seed: int = 0,
    top: int = 25,
) -> pd.DataFrame:
    """Drive ``stimulate`` and list the most active non-stimulated neurons.

    Reproduces the logic of Shiu et al. 2024 (e.g. sugar GRNs -> MN9 feeding motor
    neuron).
    """
    net = LIFNetwork(c, seed=seed)
    net.set_poisson(stimulate, rate_hz)
    rec = net.run(duration_ms)
    rates = rec.rates(c.n_neurons, duration_ms)
    rates[np.asarray(stimulate)] = 0.0
    order = np.argsort(rates)[::-1][:top]
    cols = [k for k in ("cell_type", "super_class", "cell_class", "side") if k in c.neurons]
    out = c.neurons.iloc[order][["root_id", *cols]].copy()
    out["rate_hz"] = rates[order]
    return out[out["rate_hz"] > 0].reset_index(drop=True)


def odor_lateralization(
    c: Connectome,
    rate_hz: float = 150.0,
    duration_ms: float = 500.0,
    n_trials: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Does the wiring alone turn a left/right odor difference into a DN asymmetry?

    Conditions: left ORNs only, right ORNs only, both sides. For each, the mean
    firing rate of left and right descending neurons and the lateralisation index
    ``(L - R) / (L + R)``.
    """
    orn_l, orn_r = _side_groups(c, "cell_class", "olfactory")
    dn_l, dn_r = descending_sides(c)
    net = LIFNetwork(c, seed=seed)
    rows = []
    for name, stim in (("left", orn_l), ("right", orn_r), ("both", np.r_[orn_l, orn_r])):
        for trial in range(n_trials):
            net.reset()
            net.set_poisson(stim, rate_hz)
            rates = net.run(duration_ms).rates(c.n_neurons, duration_ms)
            left, right = rates[dn_l].mean(), rates[dn_r].mean()
            rows.append(
                {
                    "stimulus": name,
                    "trial": trial,
                    "dn_left_hz": left,
                    "dn_right_hz": right,
                    "lateralization": (left - right) / (left + right + 1e-9),
                    "active_neurons": int((rates > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def ablation(
    c: Connectome,
    stimulate,
    readout,
    silence,
    rate_hz: float = 150.0,
    duration_ms: float = 1000.0,
    seed: int = 0,
) -> dict[str, float]:
    """Readout rate with and without silencing ``silence`` (in-silico lesion)."""
    res = {}
    for label, lesion in (("intact", None), ("silenced", silence)):
        net = LIFNetwork(c, seed=seed)
        if lesion is not None:
            net.silence(lesion)
        net.set_poisson(stimulate, rate_hz)
        res[label] = float(net.run(duration_ms).rates(c.n_neurons, duration_ms)[readout].mean())
    return res
