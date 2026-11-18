import numpy as np
import pandas as pd
import pytest

from flybrain.closedloop import SteeringReadout
from flybrain.data import Connectome


def two_community_graph(n=200, seed=0):
    """Two classes wired differently: A->A dense, B->A sparse, B->B dense."""
    rng = np.random.default_rng(seed)
    labels = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    pre, post = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = 0.15 if labels[i] == labels[j] else 0.01
            if rng.random() < p:
                pre.append(i)
                post.append(j)
    w = rng.integers(1, 20, len(pre)) * np.where(labels[pre] == "A", 1, -1)
    neurons = pd.DataFrame({"root_id": np.arange(n), "super_class": labels})
    return Connectome.from_edges(n, pre, post, w, neurons)


def test_classifier_separates_wiring_communities():
    from flybrain.classify import train_classifier

    rep = train_classifier(two_community_graph(), k=4, use_nt=False, min_class_size=5)
    assert rep.macro_f1 > 0.9
    assert rep.baseline_macro_f1 < 0.5


def test_steering_readout_sign():
    ro = SteeringReadout(dn_idx=np.array([0, 1]), weights=np.array([1.0, -1.0]))
    assert ro(np.array([10, 0, 5])) == pytest.approx(1.0)
    assert ro(np.array([0, 10, 5])) == pytest.approx(-1.0)
    assert ro(np.array([5, 5, 5])) == pytest.approx(0.0)
    assert ro(np.zeros(3)) == 0.0


def test_shuffled_preserves_degrees():
    from flybrain.train import shuffled

    c = two_community_graph(60)
    s = shuffled(c, seed=3)
    np.testing.assert_array_equal(
        np.bincount(c.pre, minlength=60), np.bincount(s.pre, minlength=60)
    )
    np.testing.assert_array_equal(
        np.bincount(c.post, minlength=60), np.bincount(s.post, minlength=60)
    )
    assert not np.array_equal(c.post, s.post)


def test_connectome_rnn_learns_toy_task():
    torch = pytest.importorskip("torch")
    from flybrain.train import train_steering

    torch.set_num_threads(1)
    c = two_community_graph(40, seed=1)
    res = train_steering(c, np.arange(0, 5), np.arange(20, 25), np.arange(30, 40), epochs=60)
    assert res.test_r2 > 0.5
