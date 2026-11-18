"""Cell-class prediction from wiring alone.

Question: *can we tell what a neuron is (sensory, optic, central, descending...) only
from how it is connected?* If yes, connectivity carries cell identity, which is the
premise of connectome-based cell typing.

Features per neuron (no use of the label itself):

* ``structure``  – in/out degree, in/out synapse counts, excitatory/inhibitory balance
* ``embedding``  – spectral embedding: truncated SVD of the (log-scaled) adjacency
  matrix, i.e. a low-rank summary of "who talks to whom"
* ``nt``         – predicted neurotransmitter (one-hot), optional
* ``position``   – soma / node coordinates, optional (strong but "non-connectomic")

Model: gradient-boosted trees (``HistGradientBoostingClassifier``), stratified
train/test split, class-balanced sample weights, macro-F1 as main metric because the
classes are very unbalanced (77k optic vs 110 motor neurons).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .data import Connectome
from .graph import degree_table


def spectral_embedding(c: Connectome, k: int = 16, seed: int = 0) -> np.ndarray:
    """``2k`` features: left and right singular vectors of log(1 + |W|)."""
    a = abs(c.matrix("csr")).astype(np.float64)
    a.data = np.log1p(a.data)
    k = min(k, min(a.shape) - 2)
    rng = np.random.default_rng(seed)
    u, s, vt = svds(a, k=k, v0=rng.random(min(a.shape)))
    order = np.argsort(s)[::-1]
    u, s, vt = u[:, order], s[order], vt[order]
    # rows of a = post neurons (inputs), columns = pre (outputs)
    return np.hstack([u * np.sqrt(s), vt.T * np.sqrt(s)])


def build_features(
    c: Connectome,
    k: int = 16,
    use_nt: bool = True,
    use_position: bool = False,
) -> pd.DataFrame:
    deg = degree_table(c)
    feats = deg[["in_degree", "out_degree", "in_synapses", "out_synapses", "net_input"]].astype(
        float
    )
    feats = np.log1p(feats.clip(lower=0)).assign(
        net_input_sign=np.sign(deg["net_input"]),
        out_in_ratio=np.log1p(deg["out_synapses"]) - np.log1p(deg["in_synapses"]),
    )
    emb = spectral_embedding(c, k=k)
    for i in range(emb.shape[1]):
        feats[f"svd_{i}"] = emb[:, i]
    if use_nt and "top_nt" in c.neurons:
        nt = pd.get_dummies(c.neurons["top_nt"].fillna("unknown"), prefix="nt", dtype=float)
        feats = pd.concat([feats, nt.reset_index(drop=True)], axis=1)
    if use_position:
        for col in ("pos_x", "pos_y", "pos_z"):
            if col in c.neurons:
                feats[col] = c.neurons[col].to_numpy(dtype=float)
    return feats


@dataclass
class ClassifierReport:
    accuracy: float
    macro_f1: float
    baseline_macro_f1: float
    report: str
    confusion: pd.DataFrame
    model: HistGradientBoostingClassifier

    def summary(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f}  macro-F1={self.macro_f1:.3f}  "
            f"(majority-class baseline macro-F1={self.baseline_macro_f1:.3f})"
        )


def train_classifier(
    c: Connectome,
    target: str = "super_class",
    k: int = 16,
    use_nt: bool = True,
    use_position: bool = False,
    test_size: float = 0.25,
    min_class_size: int = 20,
    seed: int = 0,
) -> ClassifierReport:
    y = c.neurons[target]
    feats = build_features(c, k=k, use_nt=use_nt, use_position=use_position)
    counts = y.value_counts()
    keep = y.isin(counts[counts >= min_class_size].index).to_numpy()
    x, y = feats[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True, random_state=seed
    )
    model.fit(x_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    pred = model.predict(x_te)

    majority = np.full(len(y_te), y_tr.value_counts().idxmax(), dtype=object)
    labels = sorted(y.unique())
    return ClassifierReport(
        accuracy=float((pred == y_te).mean()),
        macro_f1=float(f1_score(y_te, pred, average="macro")),
        baseline_macro_f1=float(f1_score(y_te, majority, average="macro")),
        report=classification_report(y_te, pred, digits=3, zero_division=0),
        confusion=pd.DataFrame(
            confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels
        ),
        model=model,
    )
