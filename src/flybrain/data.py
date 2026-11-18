"""Loading the FlyWire v783 connectome.

The raw files are not stored in this repository (≈240 MB). By default they are read
from the NeuroFly submodule (``external/NeuroFly/brain_model``); set the
``FLYBRAIN_DATA`` environment variable to point somewhere else.

Expected files (same layout as Shiu et al. 2024, *Nature*):

* ``Completeness_783.csv``     – one row per neuron, row order = Brian2 index
* ``Connectivity_783.parquet`` – one row per (pre, post) pair with synapse count and sign
* ``flywire_annotations.tsv``  – cell-type annotations (Schlegel et al. 2024)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "external" / "NeuroFly" / "brain_model"

COMPLETENESS = "Completeness_783.csv"
CONNECTIVITY = "Connectivity_783.parquet"
ANNOTATIONS = "flywire_annotations.tsv"

ANNOTATION_COLUMNS = [
    "root_id",
    "super_class",
    "cell_class",
    "cell_type",
    "side",
    "top_nt",
    "flow",
    "soma_x",
    "soma_y",
    "soma_z",
    "pos_x",
    "pos_y",
    "pos_z",
]


def data_dir() -> Path:
    """Directory holding the raw connectome files."""
    return Path(os.environ.get("FLYBRAIN_DATA", DEFAULT_DATA_DIR))


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `git submodule update --init external/NeuroFly` "
            "or set FLYBRAIN_DATA to a folder containing the FlyWire v783 files."
        )
    return path


@dataclass
class Connectome:
    """A directed, signed, weighted graph of neurons.

    Attributes
    ----------
    root_ids:
        FlyWire root id of each neuron; position in the array = neuron index.
    pre, post:
        Edge list (neuron indices).
    weight:
        Signed synapse count (``+n`` excitatory, ``-n`` inhibitory).
    neurons:
        Per-neuron annotation table aligned with ``root_ids`` (may contain NaN).
    """

    root_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    neurons: pd.DataFrame = field(repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, directory: str | Path | None = None, annotations: bool = True) -> Connectome:
        """Load the full FlyWire v783 brain (~139k neurons, ~15M connections)."""
        d = Path(directory) if directory is not None else data_dir()
        comp = pd.read_csv(_require(d / COMPLETENESS), index_col=0)
        root_ids = comp.index.to_numpy(dtype=np.int64)
        con = pd.read_parquet(
            _require(d / CONNECTIVITY),
            columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
        )
        neurons = pd.DataFrame({"root_id": root_ids})
        if annotations and (d / ANNOTATIONS).exists():
            ann = pd.read_csv(
                d / ANNOTATIONS, sep="\t", usecols=ANNOTATION_COLUMNS, low_memory=False
            ).drop_duplicates("root_id")
            neurons = neurons.merge(ann, on="root_id", how="left")
        return cls(
            root_ids=root_ids,
            pre=con["Presynaptic_Index"].to_numpy(np.int32),
            post=con["Postsynaptic_Index"].to_numpy(np.int32),
            weight=con["Excitatory x Connectivity"].to_numpy(np.float32),
            neurons=neurons,
        )

    @classmethod
    def from_edges(
        cls,
        n: int,
        pre,
        post,
        weight,
        neurons: pd.DataFrame | None = None,
    ) -> Connectome:
        """Build a connectome from an explicit edge list (useful for tests)."""
        if neurons is None:
            neurons = pd.DataFrame({"root_id": np.arange(n, dtype=np.int64)})
        return cls(
            root_ids=neurons["root_id"].to_numpy(np.int64),
            pre=np.asarray(pre, np.int32),
            post=np.asarray(post, np.int32),
            weight=np.asarray(weight, np.float32),
            neurons=neurons.reset_index(drop=True),
        )

    # ------------------------------------------------------------------ basic views
    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.weight).sum())

    def __len__(self) -> int:
        return self.n_neurons

    @cached_property
    def index_of(self) -> dict[int, int]:
        """Map FlyWire root id -> neuron index."""
        return {int(r): i for i, r in enumerate(self.root_ids)}

    def indices(self, root_ids) -> np.ndarray:
        """Neuron indices for a list of root ids (unknown ids are dropped)."""
        idx = self.index_of
        return np.array([idx[int(r)] for r in root_ids if int(r) in idx], dtype=np.int64)

    def matrix(self, fmt: str = "csr") -> sp.spmatrix:
        """Signed weight matrix ``W[post, pre]`` (so that ``W @ spikes`` = input)."""
        w = sp.coo_matrix(
            (self.weight, (self.post, self.pre)), shape=(self.n_neurons, self.n_neurons)
        )
        return w.asformat(fmt)

    def select(self, column: str, value) -> np.ndarray:
        """Indices of neurons whose annotation ``column`` equals ``value``."""
        if column not in self.neurons:
            return np.array([], dtype=np.int64)
        return np.flatnonzero(self.neurons[column].to_numpy() == value)

    # ------------------------------------------------------------------ sub-graphs
    def subnetwork(self, keep) -> Connectome:
        """Induced sub-graph on the neuron indices ``keep`` (re-indexed from 0)."""
        keep = np.unique(np.asarray(keep, dtype=np.int64))
        remap = np.full(self.n_neurons, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        mask = (remap[self.pre] >= 0) & (remap[self.post] >= 0)
        return Connectome(
            root_ids=self.root_ids[keep],
            pre=remap[self.pre[mask]].astype(np.int32),
            post=remap[self.post[mask]].astype(np.int32),
            weight=self.weight[mask],
            neurons=self.neurons.iloc[keep].reset_index(drop=True),
        )

    def neighbourhood(self, seeds, hops: int = 1, min_synapses: int = 5) -> np.ndarray:
        """Neurons reachable from ``seeds`` in ``hops`` steps (both directions).

        Only connections with at least ``min_synapses`` synapses are followed, which
        keeps the neighbourhood biologically meaningful and computationally small.
        """
        strong = np.abs(self.weight) >= min_synapses
        pre, post = self.pre[strong], self.post[strong]
        frontier = np.unique(np.asarray(seeds, dtype=np.int64))
        visited = np.zeros(self.n_neurons, dtype=bool)
        visited[frontier] = True
        for _ in range(hops):
            in_front = np.zeros(self.n_neurons, dtype=bool)
            in_front[frontier] = True
            nxt = np.concatenate([post[in_front[pre]], pre[in_front[post]]])
            nxt = np.unique(nxt)
            nxt = nxt[~visited[nxt]]
            visited[nxt] = True
            frontier = nxt
        return np.flatnonzero(visited)
