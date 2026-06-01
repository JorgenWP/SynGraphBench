"""Run GADBench cross-graph link predictors against (combined, orig)."""
from __future__ import annotations

import os
import random
import sys
import time
from typing import List

import dgl
import numpy as np
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_GADBENCH = os.path.join(_PROJECT_ROOT, 'GADBench')
_BENCH = os.path.join(_PROJECT_ROOT, 'scripts', 'benchmark')
for p in (_GADBENCH, _BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

from link_utils import LinkDataset  # noqa: E402
from models.cross_graph_link_predictor import (  # noqa: E402
    CrossGraphLinkPredictor,
    CrossGraphXGBGraphLinkPredictor,
)


LP_SUPPORTED_MODELS = ['GCN', 'GIN', 'GraphSAGE', 'XGBGraph']


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_combined_as_linkdataset(
    combined: dgl.DGLGraph,
    name: str,
    val_ratio: float,
    trial_id: int,
    neg_sampling: str,
) -> LinkDataset:
    """In-memory LinkDataset over the batched-partition combined graph.

    `test_ratio=0` because cross-graph LP tests on the original graph; the
    synthetic side only contributes train + val edges.
    """
    obj = LinkDataset.__new__(LinkDataset)
    obj.name = name
    # LinkDataset.split runs dgl.to_networkx + nx.minimum_spanning_tree
    # which require a CPU graph; cross-graph LP detectors re-`.to(device)`
    # internally, so the synthetic side stays on CPU here.
    obj.graph = combined.cpu().long()
    obj.split(val_ratio=val_ratio, test_ratio=0.0,
              trial_id=trial_id, neg_sampling=neg_sampling)
    return obj


def run_models_link(
    combined: dgl.DGLGraph,
    orig_link: LinkDataset,
    models: List[str],
    seed: int,
    device: str,
    *,
    neg_sampling: str,
    decoder: str,
    val_ratio: float,
    trial_id: int,
    epochs: int = 200,
    patience: int = 50,
    lr: float = 0.01,
    drop_rate: float = 0.0,
    h_feats: int = 32,
    num_layers: int = 2,
) -> List[dict]:
    """Run each requested LP model once on (combined, orig). One row per model.

    The synthetic-side LinkDataset is rebuilt per (seed, trial_id) so its
    train/val edge masks line up with the requested neg sampler.
    """
    rows: List[dict] = []
    for model_name in models:
        if model_name not in LP_SUPPORTED_MODELS:
            print(f"  [skip] {model_name} not in {LP_SUPPORTED_MODELS}")
            continue

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        _set_seed(seed)

        syn = wrap_combined_as_linkdataset(
            combined, name=f'combined_t{trial_id}',
            val_ratio=val_ratio, trial_id=trial_id,
            neg_sampling=neg_sampling)

        train_config = {
            'device': device,
            'epochs': epochs,
            'patience': patience,
            'metric': 'AUPRC',
            'neg_sampling': neg_sampling,
            'decoder': decoder,
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
            det = CrossGraphXGBGraphLinkPredictor(
                train_config, model_config, syn, orig_link)
        else:
            det = CrossGraphLinkPredictor(
                train_config, model_config, syn, orig_link)

        st = time.time()
        score = det.train()
        elapsed = time.time() - st

        rows.append({
            'model': model_name,
            'seed': seed,
            'AUROC': float(score['AUROC']) if score is not None else float('nan'),
            'AUPRC': float(score['AUPRC']) if score is not None else float('nan'),
            'RecK':  float(score['RecK'])  if score is not None else float('nan'),
            'time_sec': elapsed,
        })
        del det, syn

    return rows
