import numpy as np
import pytest

from flybrain.data import Connectome
from flybrain.lif import LIFNetwork, LIFParams, simulate


def chain(n=3, w=200):
    """0 -> 1 -> 2 ... strong excitatory chain."""
    return Connectome.from_edges(n, np.arange(n - 1), np.arange(1, n), np.full(n - 1, w))


def test_rest_is_a_fixed_point():
    net = LIFNetwork(chain())
    spikes = net.run(50)
    assert len(spikes) == 0
    np.testing.assert_allclose(net.v, LIFParams().v_0)


def test_exact_integration_matches_analytic_solution():
    """Free (sub-threshold) relaxation of v and g follows the closed-form solution."""
    p = LIFParams(v_th=1e9)  # never spikes
    net = LIFNetwork(chain(2), params=p)
    v0, g0 = -50.0, 3.0
    net.v[:] = v0
    net.g[:] = g0
    net.run(10.0)
    t = 10.0
    a = g0 * p.tau / (p.tau - p.t_mbr)
    expected = p.v_0 + (v0 - p.v_0 - a) * np.exp(-t / p.t_mbr) + a * np.exp(-t / p.tau)
    assert net.v[0] == pytest.approx(expected, rel=1e-6)
    assert net.g[0] == pytest.approx(g0 * np.exp(-t / p.tau), rel=1e-6)


def test_poisson_drive_sets_input_rate():
    c = Connectome.from_edges(50, [], [], [])
    rates = simulate(c, np.arange(50), rate_hz=100, duration_ms=2000)
    assert rates.mean() == pytest.approx(100, rel=0.1)


def test_spike_propagates_along_chain_with_delay():
    net = LIFNetwork(chain(3))
    net.v[0] = 0.0  # force neuron 0 above threshold at the first step
    rec = net.run(20)
    first = {int(n): float(rec.times[rec.neurons == n].min()) for n in np.unique(rec.neurons)}
    assert set(first) == {0, 1, 2}
    assert first[0] < first[1] < first[2]
    assert first[1] - first[0] >= LIFParams().t_dly


def test_inhibition_blocks_firing():
    exc = Connectome.from_edges(3, [0], [2], [200])
    inh = Connectome.from_edges(3, [0, 1], [2, 2], [200, -400])
    r_exc = simulate(exc, [0], rate_hz=100, duration_ms=1000)
    r_inh = simulate(inh, [0, 1], rate_hz=100, duration_ms=1000)
    assert r_exc[2] > 20
    assert r_inh[2] < r_exc[2] / 4


def test_refractory_period_caps_rate():
    # neuron 0 is bombarded (no refractoriness for Poisson targets) and drives 1 hard:
    # neuron 1 can only fire as fast as its refractory period allows
    net = LIFNetwork(chain(2, w=5000))
    net.set_poisson([0], 5000)
    rec = net.run(100)
    isi = np.diff(rec.times[rec.neurons == 1])
    assert isi.min() >= LIFParams().t_rfc - 1e-3  # float32 spike times


def test_silence_removes_output():
    net = LIFNetwork(chain(2))
    net.silence([0])
    net.set_poisson([0], 200)
    rec = net.run(200)
    assert (rec.neurons == 1).sum() == 0
