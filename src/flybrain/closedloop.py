"""Closed-loop brain-body simulation: whole-brain LIF  <->  NeuroMechFly body.

Loop, every ``window_ms`` (25 ms, like NeuroFly):

1. **Senses -> brain.** flygym's odor sensors (2 antennae + 2 maxillary palps) give a
   left and a right odor intensity. They set the Poisson rate of left / right
   olfactory receptor neurons (ORNs). Walking speed sets the rate of ascending
   (proprioceptive) neurons.
2. **Brain.** :class:`~flybrain.lif.LIFNetwork` simulates the full FlyWire brain
   (139k neurons) for the window.
3. **Brain -> body.** A *steering readout* turns descending-neuron (DN) spike counts
   into a turning command for flygym's ``HybridTurningController`` (left/right CPG
   amplitudes).

Difference with NeuroFly: there the turn is mostly a hand-coded odor gradient
(``odor_turn = 2.5 * tanh(20 * asym)``) and DNs only add ``0.15 * lr_diff``. Here the
turn comes **only** from DN activity. The readout weights are not trained on the
task: each DN gets weight ``+1`` / ``-1`` if, in an open-loop calibration run, it
fires more when the left / right ORNs are stimulated (see
:func:`calibrate_steering`). So the left/right decision is computed by the
connectome; we only choose which output neurons to listen to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Connectome
from .experiments import _side_groups, descending_sides
from .lif import LIFNetwork


@dataclass
class LoopConfig:
    duration_s: float = 2.0
    window_ms: float = 25.0
    physics_dt: float = 1e-4
    food_pos: tuple[float, float] = (12.0, 6.0)
    spawn_heading: float = 0.0  # radians, 0 = +x
    orn_base_hz: float = 10.0  # spontaneous ORN rate
    orn_max_hz: float = 150.0  # ORN rate at saturating odor
    odor_half_sat: float = 2.0  # odor intensity giving half of the max rate
    contrast_gain: float = 30.0  # bilateral contrast enhancement (see orn_rates)
    asc_rate_hz: float = 60.0  # ascending (proprioceptive) drive while walking
    walk_amp: float = 0.8
    turn_gain: float = 1.5
    feed_dist: float = 1.5  # mm: stop when the food is reached
    camera: str = "camera_top_zoomout"
    seed: int = 0


@dataclass
class SteeringReadout:
    """Signed DN weights; ``turn = gain * w @ counts / (|w| @ counts)``."""

    dn_idx: np.ndarray
    weights: np.ndarray

    def __call__(self, counts: np.ndarray) -> float:
        c = counts[self.dn_idx]
        total = np.abs(self.weights) @ c
        return float(self.weights @ c / total) if total > 0 else 0.0


def calibrate_steering(
    c: Connectome,
    rate_hz: float = 150.0,
    duration_ms: float = 500.0,
    n_trials: int = 4,
    z_min: float = 2.0,
    top_k: int = 40,
    seed: int = 0,
) -> tuple[SteeringReadout, dict]:
    """Open-loop calibration: which DNs prefer left vs right odor?

    Stimulate left ORNs, then right ORNs, and count each DN's spikes. Spike counts
    are Poisson-like, so the difference is scored as
    ``z = (n_left - n_right) / sqrt(n_left + n_right)``. DNs with ``|z| >= z_min``
    (at most ``top_k``) get weight ``sign(z)``: positive = "odor on the left".
    """
    orn_l, orn_r = _side_groups(c, "cell_class", "olfactory")
    dn_l, dn_r = descending_sides(c)
    dn = np.r_[dn_l, dn_r]
    net = LIFNetwork(c, seed=seed)
    counts = {}
    for name, stim in (("left", orn_l), ("right", orn_r)):
        acc = np.zeros(c.n_neurons)
        for _ in range(n_trials):
            net.reset()
            net.set_poisson(stim, rate_hz)
            acc += net.run(duration_ms).counts(c.n_neurons)
        counts[name] = acc
    n_l, n_r = counts["left"][dn], counts["right"][dn]
    z = (n_l - n_r) / np.sqrt(n_l + n_r + 1.0)
    order = np.argsort(np.abs(z))[::-1][:top_k]
    order = order[np.abs(z[order]) >= z_min]
    readout = SteeringReadout(dn_idx=dn[order], weights=np.sign(z[order]))
    t = n_trials * duration_ms / 1000
    types = c.neurons["cell_type"].to_numpy() if "cell_type" in c.neurons else None
    sides = c.neurons["side"].to_numpy() if "side" in c.neurons else None
    info = {
        "n_trials": n_trials,
        "duration_ms": duration_ms,
        "dn_mean_rate_left_stim": {
            "L": n_l[: len(dn_l)].mean() / t,
            "R": n_l[len(dn_l) :].mean() / t,
        },
        "dn_mean_rate_right_stim": {
            "L": n_r[: len(dn_l)].mean() / t,
            "R": n_r[len(dn_l) :].mean() / t,
        },
        "n_selected": int(len(order)),
        "selected": [
            {
                "cell_type": None if types is None else str(types[dn[i]]),
                "side": None if sides is None else str(sides[dn[i]]),
                "rate_left_stim_hz": float(n_l[i] / t),
                "rate_right_stim_hz": float(n_r[i] / t),
                "z": float(z[i]),
            }
            for i in order
        ],
    }
    return readout, info


@dataclass
class LoopLog:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    heading: list = field(default_factory=list)
    odor_l: list = field(default_factory=list)
    odor_r: list = field(default_factory=list)
    turn: list = field(default_factory=list)
    n_spikes: list = field(default_factory=list)
    dn_spikes: list = field(default_factory=list)
    dist: list = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def run_closed_loop(
    c: Connectome,
    readout: SteeringReadout,
    cfg: LoopConfig | None = None,
    video: str | Path | None = None,
    verbose: bool = True,
) -> LoopLog:
    """Run the brain-body loop and optionally save an MP4 of the body."""
    from flygym import Fly, YawOnlyCamera
    from flygym.arena import OdorArena
    from flygym.examples.locomotion import HybridTurningController

    cfg = cfg or LoopConfig()
    food = np.array([*cfg.food_pos, 0.0])
    arena = OdorArena(
        odor_source=food[np.newaxis],
        peak_odor_intensity=np.array([[500.0]]),
        diffuse_func=lambda x: x**-2,
        marker_colors=[(0.8, 0.4, 0.1, 1.0)],
        marker_size=0.4,
    )
    fly = Fly(
        spawn_pos=(0, 0, 0.2),
        spawn_orientation=(0, 0, cfg.spawn_heading),
        contact_sensor_placements=[
            f"{leg}{seg}"
            for leg in ("LF", "LM", "LH", "RF", "RM", "RH")
            for seg in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
        ],
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
    )
    cams = []
    if video is not None:
        cams = [
            YawOnlyCamera(
                attachment_point=fly.model.worldbody,
                camera_name=cfg.camera,
                targeted_fly_names=fly.name,
                play_speed=0.25,
                fps=30,
            )
        ]
    sim = HybridTurningController(fly=fly, cameras=cams, arena=arena, timestep=cfg.physics_dt)
    obs, _ = sim.reset(seed=cfg.seed)

    net = LIFNetwork(c, seed=cfg.seed)
    orn_l, orn_r = _side_groups(c, "cell_class", "olfactory")
    asc = c.select("super_class", "ascending")
    dn_all = np.r_[descending_sides(c)]
    stim_idx = np.r_[orn_l, orn_r, asc]
    n_l, n_r = len(orn_l), len(orn_r)

    def orn_rates(o_l: float, o_r: float) -> tuple[float, float]:
        # total intensity -> saturating drive; left/right contrast -> share of that
        # drive. Antennae are ~0.5 mm apart, so raw L/R differences are a few %;
        # the gain stands for the contrast enhancement of the antennal lobe
        # (bilateral PN/LN interactions), which the LIF model does not capture.
        total = o_l + o_r
        drive = (cfg.orn_max_hz - cfg.orn_base_hz) * total / (total + 2 * cfg.odor_half_sat)
        contrast = np.tanh(cfg.contrast_gain * (o_l - o_r) / (total + 1e-9))
        return (
            cfg.orn_base_hz + drive * (1 + contrast) / 2,
            cfg.orn_base_hz + drive * (1 - contrast) / 2,
        )

    n_windows = int(round(cfg.duration_s * 1000 / cfg.window_ms))
    phys_per_window = int(round(cfg.window_ms * 1e-3 / cfg.physics_dt))
    log = LoopLog()
    speed = 1.0
    t_wall = time.time()
    for k in range(n_windows):
        odor = obs["odor_intensity"][0].reshape(2, 2)  # rows: palps, antennae; cols: L, R
        o_l, o_r = np.average(odor, axis=0, weights=[1, 9])
        r_l, r_r = orn_rates(o_l, o_r)
        rates = np.r_[
            np.full(n_l, r_l),
            np.full(n_r, r_r),
            np.full(len(asc), cfg.asc_rate_hz * speed),
        ]
        net.set_poisson(stim_idx, rates)
        spikes = net.run(cfg.window_ms)
        counts = spikes.counts(c.n_neurons)
        turn = readout(counts)  # >0: brain says "odor on the left" -> turn left

        pos = obs["fly"][0, :2]
        dist = float(np.linalg.norm(pos - food[:2]))
        if dist < cfg.feed_dist:
            action = np.zeros(2)
        else:
            # HybridTurningController: [left amp, right amp]; turning left = slower left legs
            b = cfg.turn_gain * turn
            action = np.clip([cfg.walk_amp - b, cfg.walk_amp + b], 0.1, 1.2)
        for _ in range(phys_per_window):
            obs, _, terminated, truncated, _ = sim.step(action)
            if cams:
                sim.render()
        speed = float(np.clip(np.linalg.norm(obs["fly"][1, :2]) / 10.0, 0.15, 1.0))

        log.t.append((k + 1) * cfg.window_ms / 1000)
        log.x.append(float(obs["fly"][0, 0]))
        log.y.append(float(obs["fly"][0, 1]))
        fwd = obs["fly_orientation"]
        log.heading.append(float(np.arctan2(fwd[1], fwd[0])))
        log.odor_l.append(float(o_l))
        log.odor_r.append(float(o_r))
        log.turn.append(turn)
        log.n_spikes.append(len(spikes))
        log.dn_spikes.append(int(counts[dn_all].sum()))
        log.dist.append(dist)
        if verbose and (k % 10 == 0 or k == n_windows - 1):
            el = time.time() - t_wall
            print(
                f"  t={log.t[-1]:5.2f}s pos=({log.x[-1]:5.1f},{log.y[-1]:5.1f}) "
                f"dist={dist:5.1f}mm odor L/R={o_l:6.2f}/{o_r:6.2f} turn={turn:+.2f} "
                f"spikes={len(spikes):6d} DN={log.dn_spikes[-1]:4d}  [{el:5.0f}s]",
                flush=True,
            )
        if dist < cfg.feed_dist:
            if verbose:
                print(f"  reached food at t={log.t[-1]:.2f}s", flush=True)
            break
    if cams:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        cams[0].save_video(video)
    return log
