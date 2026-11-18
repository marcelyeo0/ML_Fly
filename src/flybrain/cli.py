"""``flybrain`` command line.

Examples::

    flybrain info
    flybrain stimulate --cell-class gustatory --side right --ms 500
    flybrain validate --trials 10
    flybrain graph
    flybrain classify
    flybrain train --epochs 100
    flybrain closedloop --seconds 2 --video outputs/closedloop.mp4
    flybrain desktop            # launch the DesktopFly overlay (Electron)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .data import REPO_ROOT, Connectome


def _out(path: str | None, default: str) -> Path:
    p = Path(path or REPO_ROOT / "outputs" / default)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cmd_info(args) -> None:
    c = Connectome.load()
    print(f"neurons      {c.n_neurons:,}")
    print(f"connections  {c.n_edges:,}")
    print(f"synapses     {c.n_synapses:,}")
    print(f"inhibitory   {(c.weight < 0).mean():.1%} of connections")
    if "super_class" in c.neurons:
        print("\nsuper_class counts:")
        print(c.neurons["super_class"].value_counts(dropna=False).to_string())


def cmd_stimulate(args) -> None:
    from .experiments import sensory_activation

    c = Connectome.load()
    idx = c.select("cell_class", args.cell_class)
    if args.side:
        idx = idx[c.neurons["side"].to_numpy()[idx] == args.side]
    if args.n:
        idx = idx[: args.n]
    print(f"stimulating {len(idx)} '{args.cell_class}' neurons at {args.rate} Hz for {args.ms} ms")
    t = time.time()
    df = sensory_activation(c, idx, rate_hz=args.rate, duration_ms=args.ms, top=args.top)
    print(df.to_string())
    print(f"[{time.time() - t:.0f} s]")


def cmd_validate(args) -> None:
    from .validate import validate_sugar

    v = validate_sugar(n_trials=args.trials)
    print(v.summary())
    v.table.to_csv(_out(args.out, "validation_sugar.csv"), index=False)


def cmd_graph(args) -> None:
    from . import graph

    c = Connectome.load()
    deg = graph.degree_table(c)
    deg["pagerank"] = graph.pagerank(c)
    out = _out(args.out, "degree_table.parquet")
    deg.to_parquet(out)
    print(f"saved {out}")
    cols = ["root_id", "cell_type", "super_class", "in_degree", "out_degree", "pagerank"]
    print("\nTop-15 PageRank neurons:")
    print(deg.nlargest(15, "pagerank")[cols].to_string(index=False))
    print("\nRich club (top 1% degree):", graph.rich_club(c))
    print("\nNeurotransmitters:")
    print(graph.excitation_inhibition(c).to_string())
    print("\nSuper-class connectivity (row-normalised, pre -> post):")
    print(graph.class_connectivity(c).round(2).to_string())
    n, _ = graph.weakly_connected(c)
    print(f"\nweakly connected components: {n}")


def cmd_classify(args) -> None:
    from .classify import train_classifier

    c = Connectome.load()
    r = train_classifier(c, use_nt=not args.no_nt, use_position=args.position)
    print(r.summary())
    print(r.report)


def cmd_train(args) -> None:
    from .train import olfactory_steering_circuit, shuffled, train_steering

    c = Connectome.load()
    sub, left, right, o = olfactory_steering_circuit(c, min_synapses=args.min_synapses)
    print(f"circuit: {sub.n_neurons} neurons, {sub.n_edges} connections, {len(o)} DNs")
    results = {}
    for name, net in (("connectome", sub), ("shuffled", shuffled(sub, seed=1))):
        print(f"[{name}]")
        res = train_steering(net, left, right, o, epochs=args.epochs, verbose=True)
        print(f"  test R2 = {res.test_r2:.3f}")
        results[name] = {"test_r2": res.test_r2, "test_mse": res.test_mse, "losses": res.losses}
    out = _out(args.out, "train_steering.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"saved {out}")


def cmd_closedloop(args) -> None:
    from .closedloop import LoopConfig, calibrate_steering, run_closed_loop
    from .plots import plot_closed_loop

    c = Connectome.load()
    print("calibrating steering readout (open loop) ...")
    t = time.time()
    readout, info = calibrate_steering(c, n_trials=args.calib_trials)
    print(f"  {info['n_selected']} steering DNs selected [{time.time() - t:.0f} s]")
    for s in info["selected"][:10]:
        print(
            f"    {s['cell_type']:>10} {s['side']:>5}  L-stim {s['rate_left_stim_hz']:6.1f} Hz"
            f"  R-stim {s['rate_right_stim_hz']:6.1f} Hz  z={s['z']:+.1f}"
        )
    cfg = LoopConfig(duration_s=args.seconds, food_pos=tuple(args.food), seed=args.seed)
    video = _out(args.video, "closedloop.mp4")
    log = run_closed_loop(c, readout, cfg, video=video)
    arrays = log.as_arrays()
    np.savez(_out(None, "closedloop_log.npz"), **arrays)
    (REPO_ROOT / "outputs" / "closedloop_calibration.json").write_text(json.dumps(info, indent=1))
    fig = plot_closed_loop(arrays, cfg.food_pos, _out(args.figure, "closedloop.png"))
    print(f"saved {video} and {fig}")


def cmd_desktop(args) -> None:
    app = REPO_ROOT / "external" / "desktop-fly" / "windows"
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    if not (app / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=app, check=True)
    subprocess.run([npm, "test" if args.test else "start"], cwd=app, check=True)


def cmd_neurofly(args) -> None:
    script = REPO_ROOT / "scripts" / "run_neurofly.py"
    subprocess.run([sys.executable, str(script), "--seconds", str(args.seconds)], check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="flybrain", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="connectome statistics").set_defaults(func=cmd_info)

    p = sub.add_parser("stimulate", help="drive a sensory population, list responders")
    p.add_argument("--cell-class", default="gustatory")
    p.add_argument("--side", choices=["left", "right"])
    p.add_argument("--n", type=int, default=0, help="only the first n neurons")
    p.add_argument("--rate", type=float, default=150.0)
    p.add_argument("--ms", type=float, default=500.0)
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=cmd_stimulate)

    p = sub.add_parser("validate", help="compare with the Brian2 reference (Shiu et al.)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--out")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("graph", help="graph analytics of the whole brain")
    p.add_argument("--out")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("classify", help="predict super-class from wiring")
    p.add_argument("--no-nt", action="store_true")
    p.add_argument("--position", action="store_true")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("train", help="connectome-constrained RNN vs shuffled wiring")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-synapses", type=int, default=30)
    p.add_argument("--out")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("closedloop", help="whole-brain LIF drives the 3D fly body")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--food", type=float, nargs=2, default=[12.0, 6.0])
    p.add_argument("--calib-trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video")
    p.add_argument("--figure")
    p.set_defaults(func=cmd_closedloop)

    p = sub.add_parser("desktop", help="launch the DesktopFly overlay (Electron)")
    p.add_argument("--test", action="store_true", help="run its test suites instead")
    p.set_defaults(func=cmd_desktop)

    p = sub.add_parser("neurofly", help="run the upstream NeuroFly simulation")
    p.add_argument("--seconds", type=float, default=0.5)
    p.set_defaults(func=cmd_neurofly)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
