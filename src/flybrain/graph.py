"""Graph analytics on the connectome.

Everything works on sparse matrices so it scales to the full brain (networkx is only
used on small sub-graphs, where its algorithms are convenient).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

from .data import Connectome


def degree_table(c: Connectome) -> pd.DataFrame:
    """In/out degree (partners) and in/out strength (synapses) of every neuron."""
    a = c.matrix("csr")
    absw = abs(a)
    binary = (absw > 0).astype(np.int32)
    df = pd.DataFrame(
        {
            "root_id": c.root_ids,
            "in_degree": np.asarray(binary.sum(axis=1)).ravel(),
            "out_degree": np.asarray(binary.sum(axis=0)).ravel(),
            "in_synapses": np.asarray(absw.sum(axis=1)).ravel(),
            "out_synapses": np.asarray(absw.sum(axis=0)).ravel(),
            "net_input": np.asarray(a.sum(axis=1)).ravel(),  # excitation - inhibition
        }
    )
    for col in ("super_class", "cell_class", "cell_type", "side", "top_nt"):
        if col in c.neurons:
            df[col] = c.neurons[col].to_numpy()
    return df


def pagerank(c: Connectome, damping: float = 0.85, tol: float = 1e-10, max_iter: int = 200):
    """PageRank on the synapse-weighted graph (power iteration on sparse matrices).

    A neuron scores high when many strongly-connected neurons project onto it: a
    proxy for "integration hubs" of the brain.
    """
    a = abs(c.matrix("csc")).astype(np.float64)  # a[post, pre]
    out = np.asarray(a.sum(axis=0)).ravel()
    dangling = out == 0
    inv = np.divide(1.0, out, out=np.zeros_like(out), where=~dangling)
    m = a @ sp.diags(inv)  # column-stochastic transition matrix
    n = c.n_neurons
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = damping * (m @ r + r[dangling].sum() / n) + (1 - damping) / n
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r / r.sum()


def class_connectivity(c: Connectome, column: str = "super_class", normalize: bool = True):
    """Synapse counts between annotation classes (rows = pre, columns = post)."""
    labels = c.neurons[column].fillna("unknown").to_numpy()
    df = pd.DataFrame({"pre": labels[c.pre], "post": labels[c.post], "syn": np.abs(c.weight)})
    mat = df.pivot_table(index="pre", columns="post", values="syn", aggfunc="sum", fill_value=0)
    if normalize:
        mat = mat.div(mat.sum(axis=1), axis=0)
    return mat


def excitation_inhibition(c: Connectome, column: str = "top_nt") -> pd.DataFrame:
    """Share of outgoing synapses per neurotransmitter."""
    labels = c.neurons[column].fillna("unknown").to_numpy()
    df = pd.DataFrame({"nt": labels[c.pre], "syn": np.abs(c.weight), "sign": np.sign(c.weight)})
    out = df.groupby("nt").agg(synapses=("syn", "sum"), sign=("sign", "mean"))
    out["fraction"] = out["synapses"] / out["synapses"].sum()
    return out.sort_values("synapses", ascending=False)


def weakly_connected(c: Connectome) -> tuple[int, np.ndarray]:
    """Number of weakly connected components and the label of each neuron."""
    return connected_components(c.matrix("csr"), directed=True, connection="weak")


def hop_distance(c: Connectome, sources, targets, min_synapses: int = 5) -> np.ndarray:
    """Minimum number of synaptic hops from any of ``sources`` to each of ``targets``.

    Only connections with at least ``min_synapses`` synapses are followed. Returns
    ``inf`` for unreachable targets. Example: how many hops separate sugar receptors
    from the feeding motor neurons?
    """
    strong = np.abs(c.weight) >= min_synapses
    a = sp.csr_matrix(
        (np.ones(strong.sum()), (c.pre[strong], c.post[strong])),
        shape=(c.n_neurons, c.n_neurons),
    )  # a[pre, post] for forward paths
    d = shortest_path(a, unweighted=True, indices=np.asarray(sources), directed=True)
    return d[:, np.asarray(targets)].min(axis=0)


def to_networkx(c: Connectome, min_synapses: int = 1) -> nx.DiGraph:
    """Convert a (small!) connectome to a networkx DiGraph with node annotations."""
    g = nx.DiGraph()
    for i, row in c.neurons.iterrows():
        g.add_node(i, **{k: v for k, v in row.items() if pd.notna(v)})
    keep = np.abs(c.weight) >= min_synapses
    g.add_weighted_edges_from(
        zip(c.pre[keep].tolist(), c.post[keep].tolist(), c.weight[keep].tolist(), strict=True)
    )
    return g


def rich_club(c: Connectome, top_fraction: float = 0.01) -> dict[str, float]:
    """Edge density among the highest-degree neurons vs the whole graph.

    A ratio well above 1 indicates a "rich club": hubs preferentially wired together.
    """
    deg = degree_table(c)
    total = (deg["in_degree"] + deg["out_degree"]).to_numpy()
    k = max(2, int(len(total) * top_fraction))
    hubs = np.argsort(total)[-k:]
    is_hub = np.zeros(c.n_neurons, bool)
    is_hub[hubs] = True
    hub_edges = np.count_nonzero(is_hub[c.pre] & is_hub[c.post])
    hub_density = hub_edges / (k * (k - 1))
    density = c.n_edges / (c.n_neurons * (c.n_neurons - 1))
    return {
        "n_hubs": k,
        "hub_density": hub_density,
        "density": density,
        "ratio": hub_density / density,
    }
