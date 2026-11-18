"""Run the NeuroFly brain-body simulation with a configurable duration.

The upstream script (``external/NeuroFly/fly_brain_body_simulation.py``) hard-codes a
10 s run, which takes ~3 h with the Brian2 C++ backend and far longer with the numpy
fallback. This wrapper patches the duration constant in memory and executes the
script unchanged otherwise, so the submodule stays pristine.

Usage::

    .venv310/Scripts/python scripts/run_neurofly.py --seconds 0.5
    .venv310/Scripts/python scripts/run_neurofly.py --seconds 0.5 --cameras camera_top_zoomout camera_back_close
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

NEUROFLY = Path(__file__).resolve().parents[1] / "external" / "NeuroFly"
SCRIPT = NEUROFLY / "fly_brain_body_simulation.py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seconds", type=float, default=0.5, help="simulated physics/brain time")
    ap.add_argument(
        "--backend",
        choices=["numpy", "cython"],
        default="numpy",
        help="Brian2 code generation target (cython needs MSVC on Windows)",
    )
    args, passthrough = ap.parse_known_args()

    src = SCRIPT.read_text(encoding="utf-8")
    src, n = re.subn(
        r"^PHYS_DURATION_S\s*=\s*[\d.]+",
        f"PHYS_DURATION_S            = {args.seconds}",
        src,
        flags=re.M,
    )
    if n != 1:
        sys.exit("could not find PHYS_DURATION_S in NeuroFly script (upstream changed?)")

    # NeuroFly prints unicode symbols; the Windows console default (cp1252) chokes on them
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    # flyvis looks for its pretrained ensemble under $FLYVIS_ROOT_DIR/results/flow;
    # NeuroFly ships the weights, so stage them there once.
    flyvis_root = NEUROFLY.parents[1] / "data" / "processed" / "flyvis"
    flow = flyvis_root / "results" / "flow" / "0000"
    if not flow.exists():
        shutil.copytree(NEUROFLY / "simulation_data" / "flyvis_pretrained_weights" / "0000", flow)
    os.environ["FLYVIS_ROOT_DIR"] = str(flyvis_root)
    from flybrain.compat import patch_datamate_windows

    patch_datamate_windows()
    os.environ["BRIAN2_NUMPY_FALLBACK"] = "1" if args.backend == "numpy" else "0"
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(NEUROFLY)
    sys.path.insert(0, str(NEUROFLY))
    sys.argv = [str(SCRIPT), *passthrough]
    code = compile(src, str(SCRIPT), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(SCRIPT)})


if __name__ == "__main__":
    main()
