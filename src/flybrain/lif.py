"""Fast whole-brain leaky integrate-and-fire (LIF) simulator.

Same model as Shiu et al. 2024 (*Nature* 634, "A Drosophila computational brain model
reveals sensorimotor processing"), which NeuroFly runs through Brian2:

.. math::

    \\tau_m \\frac{dv}{dt} = v_0 - v + g, \\qquad \\tau_s \\frac{dg}{dt} = -g

A presynaptic spike adds ``w_syn * n_syn * sign`` to ``g`` of each target after a
fixed delay. Crossing ``v_th`` emits a spike, resets ``v`` and ``g`` and makes the
neuron refractory.

Why re-implement it? Brian2 on Windows needs MSVC for its C++ backend and its numpy
fallback is ~10x slower. Here the linear ODE is integrated *exactly* per time step
(no Euler error) and synaptic propagation is one sparse column-slice per step, so the
full 139k-neuron / 15M-connection brain runs on a laptop CPU. ``tests/test_lif.py``
checks the integration against the analytic solution and ``flybrain validate``
compares firing rates with Brian2 on a sub-network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .data import Connectome


@dataclass(frozen=True)
class LIFParams:
    """Model constants (mV, ms, Hz), defaults from Shiu et al. 2024."""

    v_0: float = -52.0  # resting potential
    v_rst: float = -52.0  # reset potential
    v_th: float = -45.0  # spike threshold
    t_mbr: float = 20.0  # membrane time constant
    tau: float = 5.0  # synaptic time constant
    t_rfc: float = 2.2  # refractory period
    t_dly: float = 1.8  # synaptic delay
    w_syn: float = 0.275  # mV per synapse
    f_poi: float = 250.0  # Poisson input = w_syn * f_poi (≈ one input spike -> one output spike)


@dataclass
class SpikeRecord:
    """Spikes as two parallel arrays (time in ms, neuron index)."""

    times: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    neurons: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))

    def __len__(self) -> int:
        return len(self.times)

    def counts(self, n: int) -> np.ndarray:
        return np.bincount(self.neurons, minlength=n)

    def rates(self, n: int, duration_ms: float) -> np.ndarray:
        """Mean firing rate (Hz) of each neuron."""
        return self.counts(n) / (duration_ms / 1000.0)

    def window(self, t0: float, t1: float) -> SpikeRecord:
        m = (self.times >= t0) & (self.times < t1)
        return SpikeRecord(self.times[m], self.neurons[m])

    @staticmethod
    def concat(records: list[SpikeRecord]) -> SpikeRecord:
        if not records:
            return SpikeRecord()
        return SpikeRecord(
            np.concatenate([r.times for r in records]),
            np.concatenate([r.neurons for r in records]),
        )


class LIFNetwork:
    """Stateful LIF network: call :meth:`run` repeatedly for closed-loop control.

    Parameters
    ----------
    connectome:
        A :class:`~flybrain.data.Connectome` or a signed sparse matrix ``W[post, pre]``
        in synapse counts.
    params:
        Model constants.
    dt:
        Integration step in ms (0.1 ms like Brian2's default).
    seed:
        Seed of the Poisson input generator.
    """

    def __init__(
        self,
        connectome: Connectome | sp.spmatrix,
        params: LIFParams | None = None,
        dt: float = 0.1,
        seed: int | None = 0,
    ) -> None:
        self.p = params or LIFParams()
        self.dt = float(dt)
        w = (
            connectome.matrix("csc")
            if isinstance(connectome, Connectome)
            else sp.csc_matrix(connectome)
        )
        # weights in mV; column j = outgoing synapses of neuron j
        self.W = (w * self.p.w_syn).astype(np.float32).tocsc()
        self.n = self.W.shape[0]
        self.rng = np.random.default_rng(seed)

        # exact propagators for one step of the linear (v, g) system
        p, dt = self.p, self.dt
        self._e_m = np.exp(-dt / p.t_mbr)
        self._e_s = np.exp(-dt / p.tau)
        a = p.tau / (p.tau - p.t_mbr)  # particular-solution coefficient
        self._c = a * (self._e_s - self._e_m)  # effect of g on v over one step
        self._delay_steps = max(1, round(p.t_dly / dt))
        # Brian2: integration resumes once timestep(t - lastspike) >= timestep(t_rfc),
        # i.e. the neuron is frozen for the round(t_rfc/dt) - 1 steps after its spike
        self._rfc_steps = max(0, round(p.t_rfc / dt) - 1)

        self.poisson_idx = np.empty(0, np.int64)
        self.poisson_rate = np.empty(0, np.float64)
        self._poisson_p = np.empty(0, np.float64)
        self._rfc_exempt = np.zeros(self.n, bool)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Back to rest: ``v = v_0``, no conductance, nothing in flight."""
        self.t = 0.0
        self.step_count = 0
        self.v = np.full(self.n, self.p.v_0, np.float64)
        self.g = np.zeros(self.n, np.float64)
        self._rfc_left = np.zeros(self.n, np.int32)  # remaining refractory steps
        self._frozen = np.empty(0, np.int64)  # neurons currently refractory
        # ring buffer of future synaptic input, one row per step of delay
        self._ring = np.zeros((self._delay_steps + 1, self.n), np.float64)
        if not hasattr(self, "_rfc_exempt"):
            self._rfc_exempt = np.zeros(self.n, bool)

    def set_poisson(self, indices, rate_hz) -> None:
        """Drive ``indices`` with independent Poisson inputs at ``rate_hz``.

        Like Shiu et al., driven neurons have no refractory period so that the input
        rate maps almost one-to-one onto their output rate. ``rate_hz`` may be a
        scalar or one value per index; call again to change the drive.
        """
        idx = np.asarray(indices, np.int64)
        rate = np.broadcast_to(np.asarray(rate_hz, np.float64), idx.shape).copy()
        self.poisson_idx, self.poisson_rate = idx, rate
        self._poisson_p = rate * self.dt * 1e-3  # spike probability per step
        self._rfc_exempt[:] = False
        self._rfc_exempt[idx[rate > 0]] = True

    def silence(self, indices) -> None:
        """Remove all outgoing synapses of ``indices`` (in-silico ablation)."""
        w = self.W
        for j in np.asarray(indices, np.int64):
            w.data[w.indptr[j] : w.indptr[j + 1]] = 0.0
        w.eliminate_zeros()

    # ------------------------------------------------------------------ dynamics
    def step(self) -> np.ndarray:
        """Advance one ``dt``; return indices of neurons that spiked."""
        p = self.p
        v, g = self.v, self.g
        # refractory neurons are few: remember them and restore after the bulk update
        frozen = self._frozen
        if len(frozen):
            v_frozen, g_frozen = v[frozen], g[frozen]

        # 1) exact update of the linear (v, g) system, in place:
        #    v <- v0 + (v - v0) e_m + a g (e_s - e_m) ;  g <- g e_s
        v -= p.v_0
        v *= self._e_m
        v += self._c * g
        v += p.v_0
        g *= self._e_s
        if len(frozen):
            v[frozen], g[frozen] = v_frozen, g_frozen

        # Same order as Brian2's default schedule:
        #   groups (state update) -> thresholds -> synapses -> resets
        # 2) threshold (refractory neurons cannot fire)
        spiked = np.flatnonzero(v > p.v_th)
        if len(frozen) and len(spiked):
            spiked = spiked[self._rfc_left[spiked] == 0]

        # 3) delayed synaptic input arriving now + Poisson input (Brian2 "synapses" slot)
        slot = self.step_count % len(self._ring)
        g += self._ring[slot]
        self._ring[slot] = 0.0
        if len(self.poisson_idx):
            hits = self.rng.random(len(self.poisson_idx)) < self._poisson_p
            v[self.poisson_idx[hits]] += p.w_syn * p.f_poi

        # release neurons whose refractory period is over
        if len(frozen):
            self._rfc_left[frozen] -= 1
            self._frozen = frozen[self._rfc_left[frozen] > 0]
        # 4) reset (input that arrived this step to a spiking neuron is discarded)
        if len(spiked):
            v[spiked] = p.v_rst
            g[spiked] = 0.0
            new_ref = spiked[~self._rfc_exempt[spiked]]
            if self._rfc_steps and len(new_ref):
                self._rfc_left[new_ref] = self._rfc_steps
                self._frozen = np.union1d(self._frozen, new_ref)
            # 5) schedule postsynaptic input after the delay (CSC column gather)
            ip, ix, dat = self.W.indptr, self.W.indices, self.W.data
            starts, ends = ip[spiked], ip[spiked + 1]
            lens = ends - starts
            if lens.sum():
                sel = np.repeat(ends - lens.cumsum(), lens) + np.arange(lens.sum())
                target = (self.step_count + self._delay_steps) % len(self._ring)
                self._ring[target] += np.bincount(ix[sel], weights=dat[sel], minlength=self.n)

        self.step_count += 1
        self.t += self.dt
        return spiked

    def run(self, duration_ms: float) -> SpikeRecord:
        """Simulate ``duration_ms`` and return the spikes emitted in that window."""
        n_steps = round(duration_ms / self.dt)
        times, neurons = [], []
        for _ in range(n_steps):
            t = self.t
            s = self.step()
            if len(s):
                times.append(np.full(len(s), t, np.float32))
                neurons.append(s.astype(np.int32))
        if not times:
            return SpikeRecord()
        return SpikeRecord(np.concatenate(times), np.concatenate(neurons))


def simulate(
    connectome: Connectome,
    stimulate,
    rate_hz: float = 150.0,
    duration_ms: float = 1000.0,
    n_trials: int = 1,
    seed: int = 0,
    params: LIFParams | None = None,
) -> np.ndarray:
    """Mean firing rate (Hz) of every neuron when ``stimulate`` is Poisson-driven.

    This reproduces the core experiment of Shiu et al.: activate a sensory population
    and read out which downstream neurons respond.
    """
    net = LIFNetwork(connectome, params=params, seed=seed)
    counts = np.zeros(net.n)
    for _ in range(n_trials):
        net.reset()
        net.set_poisson(stimulate, rate_hz)
        counts += net.run(duration_ms).counts(net.n)
    return counts / n_trials / (duration_ms / 1000.0)
