"""BiGG capacity benchmark orchestrator.

Sweeps (dataset × method × partition_size) calibration trials for BiGG and
extrapolates how many training subgraphs would fit in a 1-hour wall-clock
budget at the same partition size and density. Each trial runs as an isolated
subprocess so that CUDA OOM in one trial cannot kill the sweep.

Output: one CSV row per trial, with status, train/gen seconds, peak VRAM, avg
partition (n, m, density), and `extrap_K_at_*ep` columns. Resumable — rows
already present in the CSV (matched on dataset/method_tag/partition_size) are
skipped on re-run.

Usage:
    python experiments/bigg_capacity/capacity_benchmark.py \\
        --datasets tolokers,reddit,weibo,questions,amazon,yelp \\
        --partition_sizes 500,1500,4000,10000 \\
        --methods metis,ff_b0.5_M2 \\
        --epochs 5 \\
        --num_train_subgraphs 5 \\
        --num_gen_subgraphs 1 \\
        --target_full_epochs 50,300 \\
        --time_budget_sec 3600 \\
        --csv_out experiments/bigg_capacity/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional


# Forest fire method ids parse as `ff_b{burn}_M{cap}`, where burn is a float in
# (0, 1] and cap is `1`, `2`, or `inf`. Examples: `ff_b0.5_M2`, `ff_b0.3_Minf`.
_FF_RE = re.compile(r'^ff_b(?P<burn>[0-9]*\.?[0-9]+)_M(?P<cap>1|2|inf)$')
_CAP_TAG_TO_FLAG = {'1': 'm1', '2': 'm2', 'inf': 'minf'}

# Add bigg/ to path so we can use dgl.load_graphs without entering the bigg env's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))


CSV_COLUMNS = [
    'dataset', 'method_tag', 'subsample_method', 'multiplicity_cap',
    'partition_size_target', 'K',
    'num_train_subgraphs', 'num_gen_subgraphs',
    'avg_nodes', 'avg_edges', 'avg_density',
    'train_seconds', 'gen_seconds', 'peak_vram_mb',
    'status', 'failure_phase', 'error_msg',
    'extrap_K_at_50ep', 'extrap_K_at_300ep',
]


def _load_graph_node_count(dataset: str) -> int:
    """Load the DGL graph for `dataset` and return its node count.
    Imported lazily so the orchestrator can run without the bigg env if all
    datasets are cached. Cached on disk under .capacity_cache/.
    """
    cache_path = os.path.join(_HERE, '.capacity_cache', f'{dataset}_n.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)['n']

    import dgl
    graph_path = os.path.join(_PROJECT_ROOT, 'datasets', 'original', dataset)
    graphs, _ = dgl.load_graphs(graph_path)
    n = graphs[0].num_nodes()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump({'n': n}, f)
    return n


def _completed_keys(csv_path: str) -> set:
    """Return set of (dataset, method_tag, partition_size) tuples already in the CSV."""
    if not os.path.exists(csv_path):
        return set()
    out = set()
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row['dataset'], row['method_tag'], int(row['partition_size_target'])))
    return out


def _append_row(csv_path: str, row: dict) -> None:
    """Atomic append of one row to CSV (write header on first row)."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(os.path.abspath(csv_path)) or '.', suffix='.tmp') as tmp:
        if write_header:
            existing_rows = []
        else:
            with open(csv_path, newline='') as f:
                existing_rows = list(csv.DictReader(f))
        writer = csv.DictWriter(tmp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow({k: r.get(k, '') for k in CSV_COLUMNS})
        writer.writerow({k: row.get(k, '') for k in CSV_COLUMNS})
        tmp_path = tmp.name
    os.replace(tmp_path, csv_path)


def _build_method_config(method_id: str, partition_size: int, n_total: int,
                         num_train_subgraphs: int) -> Optional[dict]:
    """Translate method_id ('metis' / 'ff_b0.5_M2' / etc.) into pipeline args.

    Returns dict with keys: subsample_method, multiplicity_cap, K (num_subgraphs),
    subsample_size, burn_prob, method_tag. Returns None if config is infeasible
    for this dataset (caller logs status=skipped_oversized).
    """
    if method_id == 'metis':
        if partition_size > n_total:
            return None
        K = max(1, math.ceil(n_total / partition_size))
        return {
            'subsample_method': 'metis',
            'multiplicity_cap': 'm1',
            'K': K,
            'subsample_size': partition_size,  # ignored by metis but required by shell
            'burn_prob': 0.5,                  # ignored by metis but required by shell
            'method_tag': f'metis_K{K}',
        }
    m = _FF_RE.match(method_id)
    if m:
        burn = float(m['burn'])
        cap_tag = m['cap']                          # '1' / '2' / 'inf'
        cap_flag = _CAP_TAG_TO_FLAG[cap_tag]        # 'm1' / 'm2' / 'minf'
        K = num_train_subgraphs
        # Forest fire with multiplicity cap M needs K * ts <= M * N.
        # M=inf: no cap. M=2: factor 2. M=1: disjoint, factor 1.
        if cap_tag != 'inf':
            cap_factor = int(cap_tag)
            if K * partition_size > cap_factor * n_total:
                return None
        return {
            'subsample_method': 'forest_fire',
            'multiplicity_cap': cap_flag,
            'K': K,
            'subsample_size': partition_size,
            'burn_prob': burn,
            'method_tag': f'ff_b{burn}_M{cap_tag}_size{partition_size}',
        }
    raise ValueError(f'Unknown method_id: {method_id!r}. '
                     f'Supported: "metis", or "ff_b{{burn}}_M{{1|2|inf}}".')


def _classify_result(returncode: int, json_path: str, stderr_text: str,
                     timed_out: bool) -> dict:
    """Parse subprocess outcome into a result dict matching CSV columns."""
    if timed_out:
        return {'status': 'timeout', 'failure_phase': None,
                'error_msg': '(subprocess hit timeout)',
                'train_seconds': None, 'gen_seconds': None, 'peak_vram_mb': None,
                'avg_nodes': None, 'avg_edges': None, 'avg_density': None,
                'num_train_subgraphs': None, 'num_gen_subgraphs': None}

    if os.path.exists(json_path):
        with open(json_path) as f:
            payload = json.load(f)
        return {
            'status': payload.get('status', 'ok'),
            'failure_phase': payload.get('failure_phase'),
            'error_msg': payload.get('error_msg'),
            'train_seconds': payload.get('train_seconds'),
            'gen_seconds': payload.get('gen_seconds'),
            'peak_vram_mb': payload.get('peak_vram_mb'),
            'avg_nodes': payload.get('avg_nodes'),
            'avg_edges': payload.get('avg_edges'),
            'avg_density': payload.get('avg_density'),
            'num_train_subgraphs': payload.get('num_train_subgraphs'),
            'num_gen_subgraphs': payload.get('num_gen_subgraphs'),
        }

    # No JSON — fall back to stderr scan
    is_oom = 'CUDA out of memory' in stderr_text or 'out of memory' in stderr_text.lower()
    return {
        'status': 'oom_unstructured' if is_oom else 'crash',
        'failure_phase': None,
        'error_msg': stderr_text[-1000:] if stderr_text else f'(exit code {returncode})',
        'train_seconds': None, 'gen_seconds': None, 'peak_vram_mb': None,
        'avg_nodes': None, 'avg_edges': None, 'avg_density': None,
        'num_train_subgraphs': None, 'num_gen_subgraphs': None,
    }


def _compute_extrapolation(train_seconds: Optional[float],
                           gen_seconds: Optional[float],
                           num_epochs: int,
                           num_train_subgraphs: int,
                           num_gen_subgraphs: int,
                           target_full_epochs: int,
                           time_budget_sec: float) -> Optional[int]:
    """How many training subgraphs of this size would fit in `time_budget_sec`
    if we ran `target_full_epochs` epochs and generated 1 synthetic per training
    subgraph at full-run time?

        T_per_epoch_sg = train_seconds / (num_epochs * num_train_subgraphs)
        T_gen_per_sg   = gen_seconds / num_gen_subgraphs
        per_sg_total   = target_full_epochs * T_per_epoch_sg + T_gen_per_sg
        extrap_K       = floor(time_budget_sec / per_sg_total)
    """
    if train_seconds is None or gen_seconds is None:
        return None
    if num_epochs <= 0 or num_train_subgraphs <= 0 or num_gen_subgraphs <= 0:
        return None
    t_per_epoch_sg = train_seconds / (num_epochs * num_train_subgraphs)
    t_gen_per_sg = gen_seconds / num_gen_subgraphs
    per_sg_total = target_full_epochs * t_per_epoch_sg + t_gen_per_sg
    if per_sg_total <= 0:
        return None
    return int(time_budget_sec // per_sg_total)


def run_one_trial(*, dataset: str, method_id: str, partition_size: int, n_total: int,
                  args: argparse.Namespace) -> dict:
    """Run one capacity trial as a subprocess and return the row dict."""
    cfg = _build_method_config(method_id, partition_size, n_total, args.num_train_subgraphs)
    if cfg is None:
        # Infeasible — record placeholder row.
        sm = 'metis' if method_id == 'metis' else 'forest_fire'
        return {
            'dataset': dataset,
            'method_tag': f'{method_id}_size{partition_size}',
            'subsample_method': sm,
            'multiplicity_cap': '',
            'partition_size_target': partition_size,
            'K': '',
            'num_train_subgraphs': '',
            'num_gen_subgraphs': '',
            'avg_nodes': '', 'avg_edges': '', 'avg_density': '',
            'train_seconds': '', 'gen_seconds': '', 'peak_vram_mb': '',
            'status': 'skipped_oversized',
            'failure_phase': '', 'error_msg': '',
            'extrap_K_at_50ep': '', 'extrap_K_at_300ep': '',
        }

    log_dir = os.path.join(_HERE, 'timing_logs', dataset)
    os.makedirs(log_dir, exist_ok=True)
    timing_log = os.path.join(log_dir, f'{cfg["method_tag"]}__size{partition_size}.json')
    if os.path.exists(timing_log):
        os.remove(timing_log)

    # Build positional args for train_bigg_capacity.sh (39 positions).
    shell_args = [
        dataset,                       # 1 dataset
        '-1',                          # 2 blksize
        '1',                           # 3 batch_size
        str(args.epochs),              # 4 epochs
        '0.001',                       # 5 lr
        '256',                         # 6 embed_dim
        '0.0',                         # 7 noise_std
        '0.0',                         # 8 ss_max_prob
        '0',                           # 9 ss_start_epoch
        'False',                       # 10 bfs_preprocess
        'zscore',                      # 11 normalize
        '0.1,0.1',                     # 12 loss_weights
        'true',                        # 13 hetero_feat
        'true',                        # 14 mask_test_labels
        '-10.0',                       # 15 logvar_floor
        str(cfg['subsample_size']),    # 16 subsample_size
        str(cfg['burn_prob']),         # 17 burn_prob
        str(cfg['K']),                 # 18 num_subgraphs
        'false',                       # 19 binary_feat
        'false',                       # 20 vae_feat
        '16',                          # 21 vae_dim
        '1.0',                         # 22 kl_weight
        'false',                       # 23 cat_feat
        '32',                          # 24 n_bins
        '',                            # 25 bin_sigma
        'false',                       # 26 mdn_feat
        '8',                           # 27 mdn_components
        '-4.0',                        # 28 mdn_logsigma_floor
        'gaussian',                    # 29 mdn_base
        'none',                        # 30 kl_schedule
        '0',                           # 31 kl_anneal_epochs
        '0',                           # 32 kl_cycle_epochs
        '0.5',                         # 33 kl_ramp_ratio
        cfg['subsample_method'],       # 34 subsample_method
        cfg['multiplicity_cap'],       # 35 multiplicity_cap
        str(args.num_train_subgraphs), # 36 num_train_subgraphs
        str(args.num_gen_subgraphs),   # 37 num_gen_subgraphs
        timing_log,                    # 38 timing_log_path
        '1.0',                         # 39 recal_momentum (disabled; capacity benchmark uses static weights)
    ]

    shell_path = os.path.join(_PROJECT_ROOT, 'scripts', 'train', 'train_bigg_capacity.sh')
    cmd = ['bash', shell_path] + shell_args
    print(f'[capacity] {dataset} / {cfg["method_tag"]} / size={partition_size} → starting subprocess')
    sys.stdout.flush()

    proc_t0 = time.perf_counter()
    timed_out = False
    stderr_text = ''
    returncode = -1
    try:
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True,
                              timeout=args.subprocess_timeout_sec)
        returncode = proc.returncode
        stderr_text = proc.stderr or ''
        # Echo last lines of stderr/stdout so SLURM logs are useful.
        if proc.stdout:
            print(proc.stdout[-1000:])
        if proc.stderr:
            print(proc.stderr[-1000:], file=sys.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stderr_text = (exc.stderr.decode() if isinstance(exc.stderr, (bytes, bytearray)) else (exc.stderr or '')) or ''
        print(f'[capacity] {dataset} / {cfg["method_tag"]} / size={partition_size} → TIMEOUT '
              f'after {args.subprocess_timeout_sec}s', file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 — orchestrator must never bubble out
        stderr_text = f'{type(exc).__name__}: {exc}'
        print(f'[capacity] subprocess raised {type(exc).__name__}: {exc}', file=sys.stderr)

    elapsed = time.perf_counter() - proc_t0
    parsed = _classify_result(returncode, timing_log, stderr_text, timed_out)

    # Compute extrapolations if we have a clean run.
    extraps = {}
    for tfe in args.target_full_epochs:
        extraps[f'extrap_K_at_{tfe}ep'] = _compute_extrapolation(
            parsed.get('train_seconds'), parsed.get('gen_seconds'),
            args.epochs,
            parsed.get('num_train_subgraphs') or args.num_train_subgraphs,
            parsed.get('num_gen_subgraphs') or args.num_gen_subgraphs,
            tfe, args.time_budget_sec) if parsed['status'] == 'ok' else None

    row = {
        'dataset': dataset,
        'method_tag': cfg['method_tag'],
        'subsample_method': cfg['subsample_method'],
        'multiplicity_cap': cfg['multiplicity_cap'],
        'partition_size_target': partition_size,
        'K': cfg['K'],
        'num_train_subgraphs': parsed.get('num_train_subgraphs') or '',
        'num_gen_subgraphs': parsed.get('num_gen_subgraphs') or '',
        'avg_nodes': parsed.get('avg_nodes') or '',
        'avg_edges': parsed.get('avg_edges') or '',
        'avg_density': parsed.get('avg_density') or '',
        'train_seconds': parsed.get('train_seconds') or '',
        'gen_seconds': parsed.get('gen_seconds') or '',
        'peak_vram_mb': parsed.get('peak_vram_mb') or '',
        'status': parsed['status'],
        'failure_phase': parsed.get('failure_phase') or '',
        'error_msg': (parsed.get('error_msg') or '').replace('\n', ' ')[:500],
    }
    for k, v in extraps.items():
        row[k] = v if v is not None else ''

    print(f'[capacity] {dataset} / {cfg["method_tag"]} / size={partition_size} → '
          f'{row["status"]} (elapsed={elapsed:.1f}s, vram={row.get("peak_vram_mb")}MB, '
          f'extrap@50ep={row.get("extrap_K_at_50ep")})')
    sys.stdout.flush()
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', type=str, required=True,
                   help='Comma-separated dataset names (matching datasets/original/<name>).')
    p.add_argument('--partition_sizes', type=str, default='500,1500,4000,10000',
                   help='Comma-separated target partition sizes (nodes per subgraph).')
    p.add_argument('--methods', type=str, default='metis,ff_b0.5_M2',
                   help='Comma-separated method ids. Supported: metis, ff_b0.5_M2.')
    p.add_argument('--epochs', type=int, default=5,
                   help='Calibration trial epoch count (default: 5).')
    p.add_argument('--num_train_subgraphs', type=int, default=5,
                   help='Number of training subgraphs in the calibration trial (default: 5).')
    p.add_argument('--num_gen_subgraphs', type=int, default=1,
                   help='Number of generation subgraphs in the calibration trial (default: 1).')
    p.add_argument('--target_full_epochs', type=str, default='50,300',
                   help='Comma-separated target epoch counts to extrapolate against.')
    p.add_argument('--time_budget_sec', type=float, default=3600.0,
                   help='Wall-clock budget in seconds for the extrapolation (default: 3600).')
    p.add_argument('--subprocess_timeout_sec', type=float, default=5400.0,
                   help='Per-trial subprocess timeout (default: 5400 = 1.5h).')
    p.add_argument('--csv_out', type=str, required=True,
                   help='Output CSV path. Resumable — already-present rows are skipped.')
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    partition_sizes = sorted(int(s) for s in args.partition_sizes.split(',') if s.strip())
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    args.target_full_epochs = [int(e) for e in args.target_full_epochs.split(',') if e.strip()]

    completed = _completed_keys(args.csv_out)
    print(f'[capacity] {len(completed)} rows already in {args.csv_out}; resuming.')

    for dataset in datasets:
        try:
            n_total = _load_graph_node_count(dataset)
            print(f'[capacity] dataset {dataset}: N={n_total}')
        except Exception as exc:  # noqa: BLE001
            print(f'[capacity] dataset {dataset}: failed to load — {exc}; skipping.',
                  file=sys.stderr)
            continue

        # Track (dataset, method) pairs that have hit a hard OOM at training/generation
        # phase — skip larger partition sizes for them.
        hard_oom_methods = set()

        for partition_size in partition_sizes:
            for method_id in methods:
                # Predict the method_tag without running the trial — used for
                # CSV-resume dedup and hard-OOM skip rows.
                if method_id == 'metis':
                    method_tag_for_check = f'metis_K{max(1, math.ceil(n_total / partition_size))}'
                    sm_for_skip = 'metis'
                else:
                    ff_m = _FF_RE.match(method_id)
                    if ff_m is None:
                        raise ValueError(
                            f'Unknown method_id: {method_id!r}. '
                            f'Supported: "metis", or "ff_b{{burn}}_M{{1|2|inf}}".')
                    method_tag_for_check = f'ff_b{float(ff_m["burn"])}_M{ff_m["cap"]}_size{partition_size}'
                    sm_for_skip = 'forest_fire'

                if (dataset, method_tag_for_check, partition_size) in completed:
                    print(f'[capacity] skip {dataset}/{method_tag_for_check}/size={partition_size} '
                          f'(already in CSV).')
                    continue

                if method_id in hard_oom_methods:
                    row = {
                        'dataset': dataset,
                        'method_tag': method_tag_for_check,
                        'subsample_method': sm_for_skip,
                        'multiplicity_cap': '',
                        'partition_size_target': partition_size,
                        'K': '',
                        'num_train_subgraphs': '',
                        'num_gen_subgraphs': '',
                        'avg_nodes': '', 'avg_edges': '', 'avg_density': '',
                        'train_seconds': '', 'gen_seconds': '', 'peak_vram_mb': '',
                        'status': 'skipped_after_oom',
                        'failure_phase': '', 'error_msg': '',
                        'extrap_K_at_50ep': '', 'extrap_K_at_300ep': '',
                    }
                    _append_row(args.csv_out, row)
                    continue

                row = run_one_trial(dataset=dataset, method_id=method_id,
                                    partition_size=partition_size,
                                    n_total=n_total, args=args)
                _append_row(args.csv_out, row)

                # Auto-skip ladder: training/generation OOM for this method on this
                # dataset → larger partition sizes will OOM too. Sampling-phase
                # OOM is host-RAM and can be transient, so we don't skip.
                if row['status'] in {'oom_training', 'oom_generation', 'oom_unstructured'}:
                    hard_oom_methods.add(method_id)
                    print(f'[capacity] {dataset}/{method_id}: hard OOM at '
                          f'partition_size={partition_size}; skipping larger sizes.')

    print(f'[capacity] sweep complete. CSV: {args.csv_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
