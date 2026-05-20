"""Run GADBench cross-graph detectors against (combined, orig)."""
from __future__ import annotations

import os
import random
import sys
import time
from typing import List

import dgl
import numpy as np
import torch

# Make the existing cross-graph detector importable without editing it
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_GADBENCH = os.path.join(_PROJECT_ROOT, 'GADBench')
_BENCH = os.path.join(_PROJECT_ROOT, 'scripts', 'benchmark')
for p in (_GADBENCH, _BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.cross_graph_detector import (  # noqa: E402
    CrossGraphGNNDetector,
    CrossGraphXGBGraphDetector,
    CrossGraphXGBDetector,
)


class _Wrap:
    """Minimal stand-in for GADBench Dataset (only `.graph` is read by detectors)."""
    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_models(
    combined: dgl.DGLGraph,
    orig: dgl.DGLGraph,
    models: List[str],
    seed: int,
    device: str,
    *,
    epochs: int = 200,
    patience: int = 50,
    lr: float = 0.01,
    drop_rate: float = 0.0,
    h_feats: int = 32,
    num_layers: int = 2,
) -> List[dict]:
    """Run each requested model once. Returns one row per model."""
    syn = _Wrap(combined)
    orig_w = _Wrap(orig)

    rows: List[dict] = []
    for model_name in models:
        torch.cuda.empty_cache()
        _set_seed(seed)

        train_config = {
            'device': device,
            'epochs': epochs,
            'patience': patience,
            'metric': 'AUPRC',
            'seed': seed,
        }
        model_config = {
            'model': model_name,
            'lr': lr,
            'drop_rate': drop_rate,
            'h_feats': h_feats,
            'num_layers': num_layers,
        }

        if model_name == 'XGBGraph':
            det = CrossGraphXGBGraphDetector(train_config, model_config, syn, orig_w)
        elif model_name == 'XGBoost':
            det = CrossGraphXGBDetector(train_config, model_config, syn, orig_w)
        else:
            det = CrossGraphGNNDetector(train_config, model_config, syn, orig_w)

        st = time.time()
        score = det.train()
        elapsed = time.time() - st

        rows.append({
            'model': model_name,
            'seed': seed,
            'AUROC': float(score['AUROC']),
            'AUPRC': float(score['AUPRC']),
            'RecK':  float(score['RecK']),
            'time_sec': elapsed,
        })
        del det

    return rows
