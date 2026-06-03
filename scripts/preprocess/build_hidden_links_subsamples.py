"""Materialize edge-pruned BiGG subsamples for the hidden_links task.

For a given (dataset, subsampling_config), runs once per split and:
1. Derives the deterministic held-out test-edge set via
   GADBench/link_utils.py:LinkDataset.split(trial_id=split_id). Seeding
   (3407 + 10*trial_id) and MST protection match what the link benchmark
   and CGT already trust, so BiGG will train on exactly the same graph
   the downstream link benchmark evaluates against.
2. Saves the held-out edges (shape (E_held, 2), full-graph node ids) to
   datasets/bigg_subsamples/<dataset>/hidden_links/held_out_edges/split<id>.pt.
   This sidecar is shared across every K-anonymity level — held-out edges
   are NOT a function of K — so all anonymized BiGG runs see the same
   evaluation edges.
3. For each partition in the source subsample pickle, drops any edge
   whose endpoints map to a held-out edge in full-graph space, then
   persists the pruned payload at
   datasets/bigg_subsamples/<dataset>/hidden_links/<config>/split<id>.pkl.

Idempotent: skips outputs whose files already exist (use --force to
overwrite).

Usage:
    python scripts/preprocess/build_hidden_links_subsamples.py \\
        --dataset tolokers --subsampling_config uniform_ts2500_K15 \\
        --splits 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import torch


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'GADBench'))
sys.path.insert(0, str(_PROJECT_ROOT / 'experiments' / 'subsample_search'))

from link_utils import LinkDataset  # noqa: E402
from splitsource import load_split_source  # noqa: E402


def _compute_held_out_edges(dataset: str, split_id: int) -> torch.Tensor:
    """Deterministic test-edge tensor for (dataset, split_id).

    Returns (E_held, 2) long tensor of (src, dst) in full-graph node ids.
    Matches the link benchmark's LinkDataset evaluation exactly (val edges
    are not held out for BiGG — only test edges, since val is purely a
    downstream model-selection set).
    """
    ds = LinkDataset(name=dataset,
                     prefix=str(_PROJECT_ROOT / 'datasets' / 'original') + '/')
    ds.split(trial_id=split_id)
    return ds.test_pos_edges.clone().long()


def _build_held_out_set(test_edges: torch.Tensor) -> set:
    """Frozen set of undirected (min,max) tuples for O(1) membership tests."""
    a = test_edges.numpy()
    return {(int(min(u, v)), int(max(u, v))) for u, v in a}


def _prune_one_partition(sg_nx: nx.Graph, orig_idx_lcc: torch.Tensor,
                         orig_ids_full: torch.Tensor,
                         held_set: set) -> tuple[nx.Graph, int]:
    """Return a copy of sg_nx with held-out edges removed; also the count dropped.

    Preserves the original graph TYPE (MultiGraph vs Graph) so the
    downstream BiGG load path's is_multigraph() simplification sees the
    same shape it always has.
    """
    cls = type(sg_nx)
    pruned = cls()
    pruned.add_nodes_from(sg_nx.nodes())
    full_for_local = orig_ids_full[orig_idx_lcc.long()].tolist()
    dropped = 0
    for u, v in sg_nx.edges():
        fu = full_for_local[u]
        fv = full_for_local[v]
        key = (fu, fv) if fu <= fv else (fv, fu)
        if key in held_set:
            dropped += 1
            continue
        pruned.add_edge(u, v)
    return pruned, dropped


def _process_split(dataset: str, subsampling_config: str, split_id: int,
                   force: bool) -> None:
    out_root = _PROJECT_ROOT / 'datasets' / 'bigg_subsamples' / dataset / 'hidden_links'
    held_dir = out_root / 'held_out_edges'
    held_path = held_dir / f'split{split_id}.pt'
    pruned_dir = out_root / subsampling_config
    pruned_path = pruned_dir / f'split{split_id}.pkl'

    # 1. held_out_edges sidecar (shared across K-anon levels)
    if not held_path.exists() or force:
        held_dir.mkdir(parents=True, exist_ok=True)
        edges = _compute_held_out_edges(dataset, split_id)
        torch.save(edges, held_path)
        print(f'  [held-out] split={split_id}: '
              f'{edges.shape[0]} edges -> {held_path}')
    else:
        print(f'  [held-out] split={split_id}: already present ({held_path})')

    # 2. Pruned partitions for the requested config
    if pruned_path.exists() and not force:
        print(f'  [pruned]   split={split_id}: already present ({pruned_path})')
        return

    src_path = (_PROJECT_ROOT / 'datasets' / 'bigg_subsamples'
                / dataset / subsampling_config / f'split{split_id}.pkl')
    if not src_path.exists():
        raise FileNotFoundError(
            f'Source subsample pickle not found: {src_path}. '
            f'Generate it via experiments/subsample_search/run_grid.py first.')

    with open(src_path, 'rb') as f:
        payload = pickle.load(f)

    held_set = _build_held_out_set(torch.load(held_path))
    src_source = load_split_source(
        dataset, split_id,
        data_dir=str(_PROJECT_ROOT / 'datasets' / 'original'))
    orig_ids_full = src_source.orig_ids.long()

    pruned_partitions = []
    total_dropped = 0
    total_edges = 0
    for sg_nx, sub_nd, orig_idx_lcc in payload['partitions']:
        before = sg_nx.number_of_edges()
        pruned_sg, dropped = _prune_one_partition(
            sg_nx, orig_idx_lcc, orig_ids_full, held_set)
        total_dropped += dropped
        total_edges += before
        pruned_partitions.append((pruned_sg, sub_nd, orig_idx_lcc))

    # Preserve everything else in the payload (meta, etc.); extend meta with
    # provenance so a stale file is easy to spot.
    new_payload = dict(payload)
    new_payload['partitions'] = pruned_partitions
    new_meta = dict(payload.get('meta', {}))
    new_meta.update({
        'task_variant': 'hidden_links',
        'held_out_edges_source': 'GADBench.LinkDataset.split',
        'held_out_edges_seed': 3407 + 10 * split_id,
        'held_out_edges_path': str(held_path.relative_to(_PROJECT_ROOT)),
        'edges_dropped': int(total_dropped),
        'edges_total_pre_prune': int(total_edges),
    })
    new_payload['meta'] = new_meta

    pruned_dir.mkdir(parents=True, exist_ok=True)
    with open(pruned_path, 'wb') as f:
        pickle.dump(new_payload, f)
    print(f'  [pruned]   split={split_id}: dropped {total_dropped}/'
          f'{total_edges} edges across {len(pruned_partitions)} partitions '
          f'-> {pruned_path}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--subsampling_config', required=True,
                   help='params_tag of the BiGG subsample set to prune '
                        '(e.g. uniform_ts2500_K15).')
    p.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing pruned pickles and sidecars.')
    args = p.parse_args()

    t0 = time.time()
    print(f'Pruning subsamples for dataset={args.dataset} '
          f'config={args.subsampling_config} splits={args.splits}')
    for sid in args.splits:
        _process_split(args.dataset, args.subsampling_config, sid, args.force)
    print(f'Done in {time.time() - t0:.1f}s.')


if __name__ == '__main__':
    main()
