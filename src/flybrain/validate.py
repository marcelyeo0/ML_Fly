"""Validate :mod:`flybrain.lif` against the published Brian2 model.

Shiu et al. 2024 ship spike trains from their Brian2 implementation
(``results/example/sugarR.parquet``: 21 right-hemisphere sugar gustatory receptor
neurons driven at 150 Hz, 30 trials x 1 s, FlyWire v630). We rerun the same
experiment with our simulator and compare neuron-by-neuron firing rates.

Because both models are stochastic (Poisson input), we do not expect identical
spikes, only matching rate statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Connectome, data_dir
from .lif import LIFNetwork

SUGAR_GRN = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]  # fmt: skip
MN9 = 720575940660219265  # proboscis motor neuron 9 (feeding)


def reference_rates(path=None, t_run: float = 1.0) -> pd.Series:
    """Mean firing rate per FlyWire id in the Brian2 reference run."""
    path = path or data_dir() / "results" / "example" / "sugarR.parquet"
    df = pd.read_parquet(path)
    n_trials = df["trial"].nunique()
    return df.groupby("flywire_id").size() / (n_trials * t_run)


@dataclass
class Validation:
    table: pd.DataFrame  # flywire_id, brian2_hz, flybrain_hz
    pearson: float
    spearman: float
    top_overlap: float
    mn9_brian2: float
    mn9_flybrain: float
    seconds_per_trial: float

    def summary(self) -> str:
        return (
            f"Pearson r={self.pearson:.3f}  Spearman rho={self.spearman:.3f}  "
            f"top-20 overlap={self.top_overlap:.0%}  "
            f"MN9: Brian2 {self.mn9_brian2:.1f} Hz vs flybrain {self.mn9_flybrain:.1f} Hz  "
            f"({self.seconds_per_trial:.0f} s per 1 s trial)"
        )


def validate_sugar(n_trials: int = 10, seed: int = 0, top: int = 20) -> Validation:
    import time

    c = Connectome.load(annotations=False, version="630")
    stim = c.indices(SUGAR_GRN)
    net = LIFNetwork(c, seed=seed)
    counts = np.zeros(c.n_neurons)
    t0 = time.time()
    for _ in range(n_trials):
        net.reset()
        net.set_poisson(stim, 150.0)
        counts += net.run(1000.0).counts(c.n_neurons)
    per_trial = (time.time() - t0) / n_trials
    ours = pd.Series(counts / n_trials, index=c.root_ids)

    ref = reference_rates()
    ids = ref.index.union(ours[ours > 0].index)
    table = pd.DataFrame(
        {"brian2_hz": ref.reindex(ids, fill_value=0.0), "flybrain_hz": ours.reindex(ids).fillna(0)}
    )
    table = table.drop(index=[i for i in SUGAR_GRN if i in table.index])  # stimulated
    a, b = table["brian2_hz"], table["flybrain_hz"]
    top_ref = set(a.nlargest(top).index)
    top_ours = set(b.nlargest(top).index)
    return Validation(
        table=table.rename_axis("flywire_id").reset_index(),
        pearson=float(np.corrcoef(a, b)[0, 1]),
        spearman=float(a.rank().corr(b.rank())),
        top_overlap=len(top_ref & top_ours) / top,
        mn9_brian2=float(ref.get(MN9, 0.0)),
        mn9_flybrain=float(ours.get(MN9, 0.0)),
        seconds_per_trial=per_trial,
    )
