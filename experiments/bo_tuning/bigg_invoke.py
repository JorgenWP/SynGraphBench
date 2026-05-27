"""Programmatic interface to BiGG training for the BO loop.

* ``build_save_name`` mirrors ``bigg/bigg/extension/pipeline.py:694-722``
  byte-for-byte so the BO coordinator can predict where BiGG will write
  before launching training. That lets us skip already-completed splits on
  resume.
* ``train_one_split`` shells out to ``scripts/train/train_bigg_subsample.sh``
  with the right positional args and waits for completion.

If pipeline.py's save_name builder is ever edited, this file must follow.
Verified at runtime against an existing tolokers output dir before the BO
study starts (see ``tests`` in ``verification.py``).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, Any


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bo_bigg_cache_root(dataset: str) -> str:
    """Per-dataset BO-scoped cache for BiGG outputs.

    Lives under the BO experiments tree so trial scaffolding doesn't
    accumulate inside the canonical ``datasets/synthetic/bigg/<dataset>/hidden_labels/``
    pipeline output. Shared across shared + per_split studies for a
    given dataset so the warm-start trial is trained once and reused
    across studies.
    """
    return os.path.join(PROJECT_ROOT, 'experiments', 'bo_tuning', dataset,
                        'bigg_cache')


def resolve_conda_env_python(env_name: str) -> str:
    """Return the absolute python binary for a named conda env on this host.

    Portable across machines: works wherever ``conda`` is on PATH (the
    SLURM templates load Anaconda3 before activating bigg, so this holds
    inside cluster jobs). Avoids hardcoding ``$HOME``-style paths in the
    YAML configs, which would break the moment the configs run on a host
    with a different conda prefix.
    """
    conda = shutil.which('conda')
    if conda is None:
        raise RuntimeError(
            "conda not found on PATH. The BO coordinator needs conda so it "
            "can locate the benchmark env's python binary. On IDUN this is "
            "provided by `module load Anaconda3/2024.02-1` in the SLURM "
            "template.")
    out = subprocess.check_output([conda, 'info', '--json']).decode()
    info = json.loads(out)
    sep = os.sep
    for env_path in info.get('envs', []):
        if env_path.rstrip(sep).endswith(sep + env_name):
            py = os.path.join(env_path, 'bin', 'python')
            if not os.path.exists(py):
                # Windows fallback / unusual layouts.
                py = os.path.join(env_path, 'python')
            if os.path.exists(py):
                return py
    raise RuntimeError(
        f"conda env {env_name!r} not found. "
        f"Available envs: {info.get('envs')}")


def resolve_benchmark_python(benchmark_cfg: Dict[str, Any]) -> str:
    """Pick the python binary for the benchmark subprocess.

    Resolution order:
      1. ``benchmark_cfg['python']`` if set and absolute (legacy override).
      2. ``benchmark_cfg['conda_env']`` → ``conda info --json`` lookup.
      3. fall back to the current interpreter (won't have xgboost unless
         the coordinator is already running in the benchmark env).
    """
    py = benchmark_cfg.get('python')
    if py and os.path.isabs(py) and os.path.exists(py):
        return py
    env = benchmark_cfg.get('conda_env')
    if env:
        return resolve_conda_env_python(env)
    return sys.executable


def build_save_name(fixed: Dict[str, Any], params: Dict[str, float], split_id: int,
                    n_subgraphs: int) -> str:
    """Replicate ``pipeline.py``'s save_name for a (fixed, tuned, split) triple.

    ``fixed`` carries all hyperparameters that the BO study holds constant
    (block size, batch size, epochs, normalize, vae/mdn flags, subsampling
    config, …). ``params`` carries the four tuned values (``lr``,
    ``kl_weight``, ``lw_cont``, ``lw_label``). ``n_subgraphs`` is the K of
    the loaded subsample partition (e.g. 5 for ``metis_K5``) and must match
    what BiGG would emit — the partitions are pre-generated, so the
    coordinator knows it from the subsampling_config.
    """
    norm_tag = fixed['normalize'] if fixed['normalize'] is not None else 'none'
    bfs_tag = 'bfs' if fixed['bfs_preprocess'] else 'nobfs'
    lw_tag = f"{params['lw_cont']}_{params['lw_label']}"
    hetero_tag = 'hetero' if fixed['hetero_feat'] else 'det'
    lvf_tag = f"_lvf{fixed['logvar_floor']}" if fixed['hetero_feat'] else ''
    mask_tag = '_masked' if fixed['mask_test_labels'] else ''
    bin_tag = '_binfeat' if fixed['binary_feat'] else ''
    vae_tag = (f"_vae{fixed['vae_dim']}_kl{params['kl_weight']}"
               if fixed['vae_feat'] else '')
    kl_sched = fixed['kl_schedule']
    if fixed['vae_feat'] and kl_sched == 'linear':
        kl_sched_tag = f"_klan{fixed['kl_anneal_epochs']}"
    elif fixed['vae_feat'] and kl_sched == 'cyclic':
        kl_sched_tag = (f"_klcyc{fixed['kl_cycle_epochs']}"
                        f"r{int(fixed['kl_ramp_ratio'] * 100)}")
    else:
        kl_sched_tag = ''
    cat_tag = ''  # cat_feat not in tuned config (mutually exclusive with mdn_feat)
    if fixed['mdn_feat']:
        base = 'lnmdn' if fixed['mdn_base'] == 'logit_normal' else 'mdn'
        mdn_tag = f"_{base}{fixed['mdn_components']}_lsf{fixed['mdn_logsigma_floor']}"
    else:
        mdn_tag = ''
    recal_tag = (f"_recalM{fixed['recal_momentum']}"
                 if fixed['recal_momentum'] < 1.0 else '')
    if fixed['load_subsamples']:
        sub_tag = (f"_loadsub_{fixed['subsampling_config']}"
                   f"_split{split_id}_n{n_subgraphs}")
    else:
        sub_tag = (f"_sub{n_subgraphs}_size{fixed['subsample_size']}"
                   f"_p{fixed['burn_prob']}")

    return (
        f"blksize_{fixed['blksize']}"
        f"_b_{fixed['batch_size']}"
        f"_lr_{params['lr']}"
        f"_epochs_{fixed['epochs']}"
        f"_noise_{fixed['noise_std']}"
        f"_ss_{fixed['ss_max_prob']}"
        f"_norm_{norm_tag}"
        f"_{bfs_tag}"
        f"_lw_{lw_tag}"
        f"_{hetero_tag}{lvf_tag}{mask_tag}{bin_tag}"
        f"{vae_tag}{kl_sched_tag}{cat_tag}{mdn_tag}{recal_tag}{sub_tag}"
    )


def expected_output_dir(dataset: str, save_name: str,
                        cache_root: str | None = None) -> str:
    """BiGG output directory for one save_name.

    With ``cache_root`` set, mirrors the BO override (env var
    ``BIGG_SYNTHETIC_SAVE_ROOT`` consumed by ``pipeline.py``). When
    ``None``, returns the legacy canonical path
    (``datasets/synthetic/bigg/<dataset>/hidden_labels/<save_name>``).
    """
    if cache_root is not None:
        return os.path.join(cache_root, save_name)
    return os.path.join(
        PROJECT_ROOT, 'datasets', 'synthetic', 'bigg', dataset,
        'hidden_labels', save_name)


def output_dir_is_complete(out_dir: str, n_subgraphs: int) -> bool:
    """True if ``out_dir`` already has the expected K subgraph files saved."""
    if not os.path.isdir(out_dir):
        return False
    entries = os.listdir(out_dir)
    n_have = sum(1 for f in entries
                 if f.startswith('subgraph_')
                 and not os.path.isdir(os.path.join(out_dir, f)))
    return n_have >= n_subgraphs


def _positional_args(fixed: Dict[str, Any], params: Dict[str, float],
                     split_id: int) -> list:
    """Build the 38 positional args for ``train_bigg_subsample.sh``.

    Order must match ``scripts/train/train_bigg_subsample.sh:67-104``.
    """
    return [
        fixed['dataset'],                               # 1
        str(fixed['blksize']),                          # 2
        str(fixed['batch_size']),                       # 3
        str(fixed['epochs']),                           # 4
        str(params['lr']),                              # 5
        str(fixed['embed_dim']),                        # 6
        str(fixed['noise_std']),                        # 7
        str(fixed['ss_max_prob']),                      # 8
        str(fixed['ss_start_epoch']),                   # 9
        str(fixed['bfs_preprocess']),                   # 10
        fixed['normalize'],                             # 11
        f"{params['lw_cont']},{params['lw_label']}",    # 12
        'true' if fixed['hetero_feat'] else 'false',    # 13
        'true' if fixed['mask_test_labels'] else 'false',  # 14
        str(fixed['logvar_floor']),                     # 15
        str(fixed['subsample_size']),                   # 16
        str(fixed['burn_prob']),                        # 17
        str(fixed.get('num_subgraphs', '') or ''),      # 18 — empty under load_subsamples
        'true' if fixed['binary_feat'] else 'false',    # 19
        'true' if fixed['vae_feat'] else 'false',       # 20
        str(fixed['vae_dim']),                          # 21
        str(params['kl_weight']),                       # 22
        'true' if fixed.get('cat_feat', False) else 'false',  # 23
        str(fixed.get('n_bins', 32)),                   # 24
        str(fixed.get('bin_sigma', '') or ''),          # 25
        'true' if fixed['mdn_feat'] else 'false',       # 26
        str(fixed['mdn_components']),                   # 27
        str(fixed['mdn_logsigma_floor']),               # 28
        fixed['mdn_base'],                              # 29
        fixed['kl_schedule'],                           # 30
        str(fixed['kl_anneal_epochs']),                 # 31
        str(fixed['kl_cycle_epochs']),                  # 32
        str(fixed['kl_ramp_ratio']),                    # 33
        'true' if fixed['load_subsamples'] else 'false',  # 34
        str(fixed.get('subsampling_config', '') or ''),  # 35
        str(split_id),                                  # 36
        str(fixed['recal_momentum']),                   # 37
        str(fixed.get('min_subgraph_nodes', 0)),        # 38
    ]


def train_one_split(fixed: Dict[str, Any], params: Dict[str, float], split_id: int,
                    n_subgraphs: int, log_dir: str,
                    cache_root: str | None = None,
                    stdout_fname: str = 'split.out',
                    stderr_fname: str = 'split.err') -> dict:
    """Train BiGG for one (HP, split) tuple. Skip if output already complete.

    Returns ``{'status': 'skipped'|'skipped_after_wait'|'completed'|'failed',
    'wall_sec': float, 'save_name': str, 'out_dir': str, 'cmd': str,
    'failure': str|None}``. The caller is responsible for symlinking
    ``out_dir`` into the trial bundle.

    When ``cache_root`` is provided, the BiGG subprocess writes there
    (via ``BIGG_SYNTHETIC_SAVE_ROOT``) and an exclusive flock on
    ``<out_dir>.lock`` serializes concurrent attempts on the same
    save_name. Concurrent jobs (e.g. shared + per_split[i] hitting the
    warm-start) wait on the lock, then re-check and return
    ``skipped_after_wait`` instead of re-training.
    """
    save_name = build_save_name(fixed, params, split_id, n_subgraphs)
    out_dir = expected_output_dir(fixed['dataset'], save_name, cache_root=cache_root)

    # Fast path: no need to touch the filesystem if it's already cached.
    if output_dir_is_complete(out_dir, n_subgraphs):
        return {'status': 'skipped', 'wall_sec': 0.0,
                'save_name': save_name, 'out_dir': out_dir,
                'cmd': '', 'failure': None}

    # Slow path: acquire an exclusive lock on the save_name dir so that
    # parallel BO jobs (shared vs per_split[i]) don't trample each other's
    # subgraph_*.dgl writes.
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    lock_path = out_dir + '.lock'

    t_wait_start = time.time()
    with open(lock_path, 'w') as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        wait_sec = time.time() - t_wait_start

        # A sibling process may have completed training while we waited.
        if output_dir_is_complete(out_dir, n_subgraphs):
            return {'status': 'skipped_after_wait', 'wall_sec': wait_sec,
                    'save_name': save_name, 'out_dir': out_dir,
                    'cmd': '', 'failure': None}

        os.makedirs(log_dir, exist_ok=True)
        script = os.path.join(PROJECT_ROOT, 'scripts', 'train', 'train_bigg_subsample.sh')
        args = _positional_args(fixed, params, split_id)
        cmd_str = 'bash ' + script + ' ' + ' '.join(_shquote(a) for a in args)

        env = os.environ.copy()
        if cache_root is not None:
            env['BIGG_SYNTHETIC_SAVE_ROOT'] = cache_root

        t0 = time.time()
        with open(os.path.join(log_dir, stdout_fname), 'w') as outf, \
             open(os.path.join(log_dir, stderr_fname), 'w') as errf:
            proc = subprocess.run(['bash', script] + args,
                                  stdout=outf, stderr=errf,
                                  cwd=PROJECT_ROOT, check=False, env=env)
        wall = time.time() - t0

        if proc.returncode != 0:
            return {'status': 'failed', 'wall_sec': wall,
                    'save_name': save_name, 'out_dir': out_dir,
                    'cmd': cmd_str,
                    'failure': f'train_bigg_subsample.sh exited with code {proc.returncode}'}
        if not output_dir_is_complete(out_dir, n_subgraphs):
            return {'status': 'failed', 'wall_sec': wall,
                    'save_name': save_name, 'out_dir': out_dir,
                    'cmd': cmd_str,
                    'failure': (f'training returned ok but expected output '
                                f'directory is incomplete: {out_dir}')}

        return {'status': 'completed', 'wall_sec': wall,
                'save_name': save_name, 'out_dir': out_dir,
                'cmd': cmd_str, 'failure': None}


def _shquote(s: str) -> str:
    """Minimal POSIX-safe shell quoting for the cmd string we log."""
    if not s:
        return "''"
    if all(c.isalnum() or c in '@%+=:,./-_' for c in s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"
