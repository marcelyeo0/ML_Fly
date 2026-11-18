"""Connectome-constrained training (requires ``torch``: ``pip install -e .[train]``).

Idea (Lappalainen et al. 2024, *Nature* — "connectome-constrained networks predict
neural activity across the fly visual system"): keep the *wiring diagram* and the
*sign* of every synapse from the connectome fixed, and only learn a handful of
per-connection gains. The connectome then acts as a very strong structural prior.

Here we train a rate network on a small sensorimotor task:

* **input**  – left / right olfactory receptor neurons (ORNs) driven with random
  intensities ``(i_L, i_R)``;
* **output** – a linear readout of descending neurons (DNs) must report the odor
  asymmetry ``i_L - i_R``, i.e. the steering command a fly needs to turn towards an
  odor plume.

Comparing with the same model on a *degree-preserving shuffled* connectome tells us
whether the real wiring helps beyond its sparsity statistics.

Dynamics (Euler, ``alpha = dt / tau``)::

    r_{t+1} = (1 - alpha) r_t + alpha * relu( (S ⊙ M ⊙ softplus(θ)) r_t + b + B u_t )

``M`` = connectome sparsity mask, ``S`` = fixed sign (Dale's law), ``θ`` = learned
log-gains initialised so that ``softplus(θ) ∝ synapse count``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Connectome

try:  # torch is optional
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _need_torch():
    if torch is None:
        raise ImportError("flybrain.train needs PyTorch: pip install -e .[train]")


def olfactory_steering_circuit(c: Connectome, hops: int = 2, min_synapses: int = 30):
    """Sub-network linking olfactory receptor neurons to descending neurons.

    Returns ``(sub, input_left, input_right, output)`` index arrays in ``sub``.
    """
    orn = c.select("cell_class", "olfactory")
    dn = c.select("super_class", "descending")
    near = c.neighbourhood(orn, hops=hops, min_synapses=min_synapses)
    # keep descending neurons that receive strong input from that population
    strong = (np.abs(c.weight) >= min_synapses // 3) & np.isin(c.pre, near) & np.isin(c.post, dn)
    sub = c.subnetwork(np.union1d(near, np.unique(c.post[strong])))
    side = sub.neurons["side"].fillna("").to_numpy()
    cls = sub.neurons["cell_class"].fillna("").to_numpy()
    sup = sub.neurons["super_class"].fillna("").to_numpy()
    in_l = np.flatnonzero((cls == "olfactory") & (side == "left"))
    in_r = np.flatnonzero((cls == "olfactory") & (side == "right"))
    out = np.flatnonzero(sup == "descending")
    return sub, in_l, in_r, out


def shuffled(c: Connectome, seed: int = 0) -> Connectome:
    """Degree-preserving null model: permute post-synaptic targets of the edge list.

    Keeps every neuron's out-degree, in-degree and the weight distribution, destroys
    the specific wiring.
    """
    rng = np.random.default_rng(seed)
    post = rng.permutation(c.post)
    return Connectome(c.root_ids, c.pre.copy(), post, c.weight.copy(), c.neurons)


if nn is not None:

    class ConnectomeRNN(nn.Module):
        """Rate RNN whose recurrent matrix is constrained by a connectome."""

        def __init__(
            self,
            c: Connectome,
            inputs: np.ndarray,
            outputs: np.ndarray,
            n_in: int,
            tau: float = 10.0,
            dt: float = 2.0,
            gain: float = 0.9,
        ):
            super().__init__()
            n = c.n_neurons
            idx = torch.tensor(np.stack([c.post, c.pre]), dtype=torch.long)
            self.register_buffer("idx", idx)
            self.register_buffer("sign", torch.tensor(np.sign(c.weight), dtype=torch.float32))
            # normalise each neuron's total input to ``gain`` < 1: the recurrent matrix
            # starts contractive (stable), learning can then reshape it
            absw = np.abs(c.weight).astype(np.float64)
            in_strength = np.bincount(c.post, weights=absw, minlength=n)
            g0 = np.clip(gain * absw / in_strength[c.post], 1e-6, None)
            init = np.log(np.expm1(g0))  # inverse softplus
            self.theta = nn.Parameter(torch.tensor(init, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(n))
            b_in = torch.zeros(n, n_in)
            for k, group in enumerate(inputs):
                b_in[torch.as_tensor(group), k] = 1.0
            self.register_buffer("b_in", b_in)
            self.in_gain = nn.Parameter(torch.ones(1))
            self.register_buffer("out_idx", torch.as_tensor(outputs, dtype=torch.long))
            self.readout = nn.Linear(len(outputs), 1)
            self.n, self.alpha = n, dt / tau

        def weight(self):
            vals = self.sign * nn.functional.softplus(self.theta)
            return torch.sparse_coo_tensor(
                self.idx, vals, (self.n, self.n), check_invariants=False
            ).coalesce()

        def forward(self, u: torch.Tensor, steps: int = 25) -> torch.Tensor:
            """``u``: (batch, n_in) constant input. Returns (batch,) readout."""
            w = self.weight()
            drive = (self.b_in @ u.T).T * self.in_gain  # (batch, n)
            r = torch.zeros(u.shape[0], self.n)
            for _ in range(steps):
                rec = torch.sparse.mm(w, r.T).T
                x = rec + self.bias + drive
                r = (1 - self.alpha) * r + self.alpha * torch.tanh(torch.relu(x))
            return self.readout(r[:, self.out_idx]).squeeze(-1)


@dataclass
class TrainResult:
    losses: list
    test_mse: float
    test_r2: float
    n_neurons: int
    n_edges: int
    n_params: int


def train_steering(
    c: Connectome,
    in_l,
    in_r,
    out,
    epochs: int = 150,
    batch: int = 64,
    lr: float = 0.02,
    seed: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Train :class:`ConnectomeRNN` to output ``i_L - i_R`` from ORN drive."""
    _need_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ConnectomeRNN(c, [in_l, in_r], out, n_in=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def sample(n):
        u = torch.tensor(rng.random((n, 2)), dtype=torch.float32)
        return u, u[:, 0] - u[:, 1]

    losses = []
    for epoch in range(epochs):
        u, y = sample(batch)
        loss = nn.functional.mse_loss(model(u), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if verbose and epoch % 25 == 0:
            print(f"  epoch {epoch:4d}  mse={loss.item():.4f}")

    with torch.no_grad():
        u, y = sample(512)
        pred = model(u)
        mse = float(nn.functional.mse_loss(pred, y))
        r2 = 1 - mse / float(y.var())
    return TrainResult(
        losses=losses,
        test_mse=mse,
        test_r2=r2,
        n_neurons=c.n_neurons,
        n_edges=c.n_edges,
        n_params=sum(p.numel() for p in model.parameters()),
    )
