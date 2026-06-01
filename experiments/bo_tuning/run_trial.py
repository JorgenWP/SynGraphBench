"""One BO trial: train 5 BiGGs (one per split) → build bundle → run benchmark.

In ``shared`` mode the same HPs are applied to every split; in ``per_split``
mode the BO study calling us trains only one split (the caller picks which)
and the bundle contains a single variant. Either way, the resulting
``per_trial_results.csv`` is fed to ``objective.compute_objective``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd

from .bigg_invoke import (
    train_one_split, expected_output_dir, PROJECT_ROOT,
    resolve_benchmark_python,
)
from .objective import compute_objective, base_rate_from_test_mask


BENCHMARK_SCRIPT = os.path.join(PROJECT_ROOT, 'scripts', 'benchmark',
                                'anomaly_benchmark.py')


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _git_commit():
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None


def _ensure_bundle(trial_dir, dataset, save_names_by_split, cache_root=None):
    """Materialise the trial's bundle dir with one symlink per split.

    Symlink targets are the BiGG output dirs (under ``cache_root`` when
    set, otherwise the canonical pipeline location). The link names are
    the BiGG save_names themselves, so ``discover_bigg_split_bundle``
    picks up the ``_split{N}_n{K}`` tag without any rewriting.
    """
    bundle_dir = os.path.join(trial_dir, 'bundle')
    os.makedirs(bundle_dir, exist_ok=True)
    for split_id, save_name in save_names_by_split.items():
        target = expected_output_dir(dataset, save_name, cache_root=cache_root)
        link = os.path.join(bundle_dir, save_name)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(target, link)
    return bundle_dir


def run_trial(*, dataset, fixed_hp, params, split_ids, trial_dir,
              n_subgraphs_by_split, models, seeds_per_split, tune_test_ratio,
              tune_test_seed, tune_portion, mode, benchmark_env,
              baseline_csv_path=None, cache_root=None,
              data_dir=None, semi_supervised=0):
    """Execute one BO trial end-to-end.

    Returns a dict mixing training timings, benchmark metrics, and the final
    objective. Raises only for unrecoverable orchestration errors —
    per-split or benchmark failures are reported via ``status`` in the
    returned dict so the caller can mark the Optuna trial FAILED.
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, 'datasets', 'original')

    os.makedirs(trial_dir, exist_ok=True)
    logs_dir = os.path.join(trial_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    started_at = _now_iso()
    t0 = time.time()

    split_train_sec = {}
    save_names_by_split = {}
    failure = None
    status = 'COMPLETED'

    for sid in split_ids:
        log_dir = os.path.join(logs_dir, f'split{sid}')
        res = train_one_split(
            fixed_hp, params, split_id=sid,
            n_subgraphs=n_subgraphs_by_split[sid],
            log_dir=log_dir, cache_root=cache_root,
            stdout_fname='train.out', stderr_fname='train.err')
        split_train_sec[sid] = res['wall_sec']
        save_names_by_split[sid] = res['save_name']
        if res['status'] == 'failed':
            failure = {
                'stage': f'train_split_{sid}',
                'message': res['failure'] or 'unknown',
                'cmd': res['cmd'],
            }
            status = 'FAILED'
            break

    metrics = {
        'started_at': started_at,
        'mode': mode,
        'params': params,
        'split_ids': list(split_ids),
        'split_train_sec': split_train_sec,
        'save_names': save_names_by_split,
        'status': status,
        'failure': failure,
        'git_commit': _git_commit(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'host': os.environ.get('SLURMD_NODENAME') or os.uname().nodename,
    }

    if status == 'FAILED':
        metrics['wall_sec'] = time.time() - t0
        metrics['completed_at'] = _now_iso()
        _flush_metadata(trial_dir, metrics)
        return metrics

    bundle_dir = _ensure_bundle(trial_dir, dataset, save_names_by_split,
                                cache_root=cache_root)

    # Benchmark with the new flags. Output goes inside the trial dir.
    bench_out = os.path.join(trial_dir, 'benchmark')
    os.makedirs(bench_out, exist_ok=True)
    bench_log = os.path.join(logs_dir, 'benchmark.out')
    bench_err = os.path.join(logs_dir, 'benchmark.err')

    cmd = [
        resolve_benchmark_python(benchmark_env), '-u', BENCHMARK_SCRIPT,
        '--datasets', dataset,
        '--models', ','.join(models),
        '--generator', 'bigg',
        '--synthetic_type', 'graph',
        '--task', 'hidden_labels',
        '--graph_path', bundle_dir,
        '--seeds_per_split', str(seeds_per_split),
        '--epochs', str(benchmark_env['epochs']),
        '--patience', str(benchmark_env['patience']),
        '--lr', str(benchmark_env['lr']),
        '--drop_rate', str(benchmark_env['drop_rate']),
        '--h_feats', str(benchmark_env['h_feats']),
        '--num_layers', str(benchmark_env['num_layers']),
        '--dump_per_trial',
        '--tune_test_ratio', str(tune_test_ratio),
        '--tune_test_seed', str(tune_test_seed),
        '--tune_portion', tune_portion,
        '--output_dir', bench_out,
    ]
    if semi_supervised:
        cmd += ['--semi_supervised', str(semi_supervised)]
    if baseline_csv_path is not None:
        # Coordinator has the baseline cached; per-trial benchmark only
        # needs Phase 2. We merge the cached baseline back in below before
        # computing the objective.
        cmd += ['--skip_original']

    t_bench = time.time()
    with open(bench_log, 'w') as outf, open(bench_err, 'w') as errf:
        proc = subprocess.run(cmd, stdout=outf, stderr=errf,
                              cwd=PROJECT_ROOT, check=False)
    metrics['benchmark_sec'] = time.time() - t_bench
    metrics['benchmark_cmd'] = ' '.join(cmd)

    if proc.returncode != 0:
        metrics['status'] = 'FAILED'
        metrics['failure'] = {
            'stage': 'benchmark',
            'message': f'anomaly_benchmark.py exited with code {proc.returncode}',
            'log': bench_err,
        }
        metrics['wall_sec'] = time.time() - t0
        metrics['completed_at'] = _now_iso()
        _flush_metadata(trial_dir, metrics)
        return metrics

    per_trial_csv = os.path.join(bench_out, 'per_trial_results.csv')
    if not os.path.exists(per_trial_csv):
        metrics['status'] = 'FAILED'
        metrics['failure'] = {
            'stage': 'parse_results',
            'message': f'per_trial_results.csv not produced at {per_trial_csv}',
        }
        metrics['wall_sec'] = time.time() - t0
        metrics['completed_at'] = _now_iso()
        _flush_metadata(trial_dir, metrics)
        return metrics

    per_trial_df = pd.read_csv(per_trial_csv)

    # Merge cached baseline rows in (Phase 1 was skipped per --skip_original).
    if baseline_csv_path is not None:
        baseline_df = pd.read_csv(baseline_csv_path)
        per_trial_df = pd.concat([per_trial_df, baseline_df], ignore_index=True)

    # Base rate is fixed across trials (same dataset, mask, portion).
    # Compute on the first split we actually evaluated.
    first_split = sorted(split_ids)[0]
    base_rate = base_rate_from_test_mask(
        dataset, data_dir, tune_test_ratio, tune_test_seed, tune_portion,
        semi_supervised=semi_supervised, split_id=first_split)

    obj = compute_objective(per_trial_df, base_rate=base_rate, mode=mode)
    metrics.update(obj)
    metrics['per_trial_csv'] = per_trial_csv

    # Per-(split, model, seed) AUPRC dicts for the thesis metadata schema.
    metrics['per_split_per_model_per_seed_auprc'] = _stack(per_trial_df, 'AUPRC', 'synthetic-graph')
    metrics['per_split_per_model_per_seed_auroc'] = _stack(per_trial_df, 'AUROC', 'synthetic-graph')
    metrics['per_split_per_model_per_seed_reck']  = _stack(per_trial_df, 'RecK',  'synthetic-graph')
    metrics['per_split_per_model_baseline_auprc'] = _stack(per_trial_df, 'AUPRC', 'original')

    metrics['wall_sec'] = time.time() - t0
    metrics['completed_at'] = _now_iso()
    _flush_metadata(trial_dir, metrics)
    return metrics


def _stack(df, value_col, source):
    sub = df[df['source'] == source]
    out = {}
    for split_id, g in sub.groupby('split_id'):
        out_inner = {}
        for model, gm in g.groupby('model'):
            out_inner[str(model)] = [float(v) for v in gm[value_col].tolist()]
        out[int(split_id)] = out_inner
    return out


def _flush_metadata(trial_dir, metrics):
    out_path = os.path.join(trial_dir, 'metadata.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)


if __name__ == '__main__':
    # Re-entry point so coordinators that prefer one-trial-per-process can
    # invoke this with a JSON args blob on stdin. Used by the array worker
    # SLURM template in topology C.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_json', required=True,
                        help='Path to a JSON file with the run_trial(**kwargs) blob.')
    parser.add_argument('--out_json', required=True,
                        help='Where to write the returned metrics dict.')
    cli = parser.parse_args()
    with open(cli.args_json) as f:
        kwargs = json.load(f)
    res = run_trial(**kwargs)
    with open(cli.out_json, 'w') as f:
        json.dump(res, f, indent=2, sort_keys=True, default=str)
    sys.exit(0 if res.get('status') == 'COMPLETED' else 1)
