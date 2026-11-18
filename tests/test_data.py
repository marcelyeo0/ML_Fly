import numpy as np
import pandas as pd

from flybrain.data import Connectome


def toy():
    neurons = pd.DataFrame(
        {
            "root_id": [10, 11, 12, 13],
            "super_class": ["sensory", "central", "central", "descending"],
        }
    )
    return Connectome.from_edges(4, [0, 1, 2, 0], [1, 2, 3, 3], [5, -7, 9, 1], neurons)


def test_matrix_orientation_is_post_by_pre():
    w = toy().matrix().toarray()
    assert w[1, 0] == 5
    assert w[2, 1] == -7
    assert w.shape == (4, 4)


def test_subnetwork_reindexes_and_keeps_internal_edges():
    sub = toy().subnetwork([1, 2, 3])
    assert list(sub.root_ids) == [11, 12, 13]
    assert sub.n_edges == 2
    assert sub.matrix().toarray()[1, 0] == -7


def test_neighbourhood_respects_synapse_threshold():
    c = toy()
    assert set(c.neighbourhood([0], hops=1, min_synapses=5)) == {0, 1}
    assert set(c.neighbourhood([0], hops=1, min_synapses=1)) == {0, 1, 3}


def test_select_and_indices():
    c = toy()
    np.testing.assert_array_equal(c.select("super_class", "central"), [1, 2])
    np.testing.assert_array_equal(c.indices([13, 10, 999]), [3, 0])
    assert c.n_synapses == 22
