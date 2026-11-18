# flybrain — Drosophila connectome, spiking brain and closed-loop body

Whole-brain simulation of the fruit fly *Drosophila melanogaster* built on the
FlyWire connectome (v783: 138,639 neurons, 15.1 M connections, 54.5 M synapses).

* **`flybrain.lif`** — a leaky integrate-and-fire simulator of the *entire* brain,
  same model as Shiu et al. 2024 (*Nature*), re-implemented with exact integration and
  sparse matrices so it runs on a laptop CPU without a C++ compiler.
  Validated against the published Brian2 run: **Pearson r = 0.96** on per-neuron
  firing rates.
* **`flybrain.closedloop`** — the simulated brain drives the 3D NeuroMechFly body
  (MuJoCo, via flygym): odor sensors → olfactory receptor neurons → whole brain →
  descending neurons → walking/turning.
* **`flybrain.graph`** — graph analytics (degrees, PageRank, class-to-class wiring,
  hop distances, rich club).
* **`flybrain.classify`** — predicts a neuron's super-class from its wiring alone:
  **97 % accuracy, macro-F1 0.86** (majority baseline 0.07).
* **`flybrain.train`** — connectome-constrained RNN (PyTorch): wiring and synapse
  signs fixed by the connectome, gains learned, compared with a shuffled connectome.
* **Desktop fly** — the [DesktopFly](https://github.com/DenisSergeevitch/desktop-fly)
  Electron app (a 3D fly that walks on your Windows desktop, driven by a 668-neuron
  FlyWire circuit) is included as a submodule.
* **NeuroFly** — the original [NeuroFly](https://github.com/seven-monarchs/NeuroFly)
  brain-body pipeline (Brian2 + flygym + flyvis) is included as a submodule, with a
  runner that makes it work on Windows.

📖 **Explications détaillées et guide d'entretien : [docs/PROJET.md](docs/PROJET.md)**

## Results

| experiment | result | file |
| --- | --- | --- |
| LIF vs Brian2 reference (sugar GRNs → MN9, 10 trials) | Pearson r = 0.956, Spearman 0.962, MN9 106 vs 93 Hz | `results/validation_sugar.txt` |
| super-class from wiring only (GBDT, 25 % test) | accuracy 0.973, macro-F1 0.860 | `results/classifier_super_class.txt` |
| + neurotransmitter + position | accuracy 0.981, macro-F1 0.897 | idem |
| speed, 25 ms of whole brain (no C++ compiler) | flybrain ≈ 1–2 s vs Brian2 numpy ≈ 500 s | `docs/PROJET.md` §6.2 |
| left vs right ORN stimulation | ipsilateral bias in DNa02 (known steering DN), crossed DNbe007/DNb05 | `results/odor_lateralization.txt` |
| connectome-constrained RNN (ORN → DN steering) | test R² 0.85 after 20 epochs; shuffled-wiring control still to run | `flybrain train` |
| closed loop brain ↔ body | 15 lateralised steering DNs calibrated; short sub-network run OK; 2 s whole-brain run pending | `results/closedloop_calibration.txt` |

## Install

Requirements: Windows 10/11 (Linux/macOS work too), Python 3.10, Node 18+, git, ffmpeg.

```bash
git clone --recurse-submodules https://github.com/marcelyeo0/ML_Fly.git
cd ML_Fly
uv venv --python 3.10 .venv310                    # or: py -3.10 -m venv .venv310
uv pip install --python .venv310 torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv310 -e ".[dev,sim]"
```

The connectome files (~240 MB) come with the NeuroFly submodule
(`external/NeuroFly/brain_model`). Set `FLYBRAIN_DATA` to use another folder.

## Usage

```bash
flybrain info                                   # connectome statistics
flybrain stimulate --cell-class gustatory --side right --n 21   # who responds?
flybrain validate --trials 10                   # LIF vs Brian2 reference
flybrain graph                                  # PageRank, class wiring, rich club
flybrain classify                               # cell class from wiring
flybrain train --epochs 100                     # connectome-constrained RNN
flybrain closedloop --seconds 2                 # brain drives the 3D body -> outputs/closedloop.mp4
flybrain desktop                                # the fly on your desktop (Electron)
flybrain neurofly --seconds 0.5                 # upstream NeuroFly pipeline (slow)
pytest                                          # unit tests
```

## Layout

```text
src/flybrain/
  data.py        Connectome: loading, sparse matrix, sub-networks
  lif.py         whole-brain LIF simulator (exact integration, delays, refractoriness)
  validate.py    comparison with the Brian2 reference results
  experiments.py sensory activation, odor lateralisation, ablation
  graph.py       graph analytics
  classify.py    cell-class prediction from wiring
  train.py       connectome-constrained RNN (PyTorch)
  closedloop.py  brain <-> NeuroMechFly body (flygym / MuJoCo)
  plots.py       figures
  compat.py      Windows fix for datamate (used by flyvis)
  cli.py         `flybrain` command
scripts/run_neurofly.py   runs upstream NeuroFly with a short duration on Windows
external/desktop-fly      submodule — Electron desktop fly
external/NeuroFly         submodule — original brain-body pipeline + data
results/                  small result files quoted in the docs
docs/PROJET.md            full explanation (French) + interview guide
```

## Data and credits

FlyWire connectome (Dorkenwald et al. 2024, Schlegel et al. 2024, *Nature*) —
CC-BY 4.0. LIF model and reference results: Shiu et al. 2024, *Nature*.
NeuroMechFly / flygym: Wang-Chen et al. 2024, *Nature Methods*. flyvis: Lappalainen
et al. 2024, *Nature*. NeuroFly by seven-monarchs; DesktopFly by DenisSergeevitch
(MIT). This repository's code: MIT.
