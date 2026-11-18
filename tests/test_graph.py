import numpy as np
import pandas as pd
import pytest

from flybrain import graph
from flybrain.data import Connectome


@pytest.fixture
def toy():
    neurons = pd.DataFrame(
        {
            "root_id": [10, 11, 12, 13],
            "super_class": ["sensory", "central", "central", "descending"],
            "top_nt": ["acetylcholine", "gaba", "acetylcholine", "acetylcholine"],
        }
    )
    return Connectome.from_edges(4, [0, 1, 2, 0], [1, 2, 3, 3], [5, -7, 9, 1], neurons)


def test_degree_table(toy):
    d = graph.degree_table(toy)
    assert d.loc[3, "in_degree"] == 2
    assert d.loc[0, "out_synapses"] == 6
    assert d.loc[2, "net_input"] == -7


def test_pagerank_sums_to_one_and_ranks_sink_high(toy):
    r = graph.pagerank(toy)
    assert r.sum() == pytest.approx(1.0)
    assert r.argmax() == 3


def test_class_connectivity_rows_normalised(toy):
    m = graph.class_connectivity(toy)
    np.testing.assert_allclose(m.sum(axis=1), 1.0)
    assert m.loc["sensory", "central"] == pytest.approx(5 / 6)


def test_hop_distance(toy):
    d = graph.hop_distance(toy, [0], [1, 2, 3], min_synapses=1)
    np.testing.assert_array_equal(d, [1, 2, 1])
    d_strong = graph.hop_distance(toy, [0], [3], min_synapses=5)
    assert d_strong[0] == 3


def test_networkx_and_components(toy):
    g = graph.to_networkx(toy)
    assert g.number_of_edges() == 4
    n, _ = graph.weakly_connected(toy)
    assert n == 1
