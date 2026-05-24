"""Re-enqueue trials whose runs were killed by a SLURM walltime.

When the coordinator is killed mid-trial, Optuna leaves the trial in state
``RUNNING`` indefinitely, which blocks the study's progress accounting.
At every coordinator startup we call this to mark stale RUNNING trials as
FAILED and re-enqueue their params so TPE doesn't have to rediscover.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import optuna


def cleanup_stale_running(study, stale_after_sec):
    """Mark any RUNNING trials older than the threshold as FAILED + re-enqueue.

    Returns the number of trials cleaned up.
    """
    now = datetime.now(timezone.utc)
    n_cleaned = 0
    for trial in study.get_trials(deepcopy=False,
                                  states=(optuna.trial.TrialState.RUNNING,)):
        # ``datetime_start`` is naive (local time). Treat as UTC for our purposes —
        # we only care about elapsed seconds; a small offset is harmless.
        start = trial.datetime_start
        if start is None:
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        elapsed = (now - start).total_seconds()
        if elapsed < stale_after_sec:
            continue

        params = dict(trial.params)
        try:
            study._storage.set_trial_state_values(
                trial._trial_id, optuna.trial.TrialState.FAIL)
        except Exception as e:
            print(f'[cleanup_stale] could not mark trial {trial.number} '
                  f'failed: {e}')
            continue
        if params:
            # skip_if_exists=False: we *just* marked this trial FAIL, and
            # optuna's duplicate check considers prior-enqueued trials in
            # FAIL/RUNNING/WAITING. With skip_if_exists=True the just-failed
            # trial would match itself and the re-enqueue would silently
            # no-op — losing the params for warm-start/manually-enqueued trials.
            study.enqueue_trial(params, skip_if_exists=False)
            print(f'[cleanup_stale] marked trial {trial.number} FAILED '
                  f'(stale for {elapsed/3600:.1f}h) and re-enqueued params.')
        else:
            print(f'[cleanup_stale] marked trial {trial.number} FAILED '
                  f'(stale for {elapsed/3600:.1f}h, no params to re-enqueue).')
        n_cleaned += 1
    return n_cleaned


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_db', required=True,
                        help='Path to the SQLite study file.')
    parser.add_argument('--study_name', required=True)
    parser.add_argument('--stale_after_sec', type=int, default=6 * 3600)
    args = parser.parse_args()
    storage = optuna.storages.RDBStorage(
        f'sqlite:///{args.study_db}',
        engine_kwargs={'connect_args': {'timeout': 30}})
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    n = cleanup_stale_running(study, args.stale_after_sec)
    print(f'cleaned {n} stale RUNNING trials.')
