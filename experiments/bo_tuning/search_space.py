"""Optuna search space for BiGG hyperparameter tuning.

Four log-uniform dimensions: learning rate, KL weight, and the two
``-loss_weights`` components (continuous-feature and label weights relative
to the structural loss). Bounds are roughly 1-2 orders of magnitude around
the per-dataset defaults known to work; the upper KL bound is capped to
avoid posterior collapse and the upper ``lw_cont`` bound is capped to avoid
structural degradation.
"""

from __future__ import annotations

# Per-dataset warm-start defaults — these reproduce the best-known config
# from prior manual sweeps, used to enqueue trial 0 so TPE anchors on a
# known-good region of the surrogate.
WARM_START_DEFAULTS = {
    'tolokers':  {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
    'questions': {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.01, 'lw_label': 0.10},
    'weibo':     {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
    'reddit':     {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
    'elliptic':     {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
    'amazon':     {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
    'tfinance':     {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
    'yelp':     {'lr': 3e-4, 'kl_weight': 0.25, 'lw_cont': 0.05, 'lw_label': 0.10},
}

# Conservative bounds. Override per dataset in configs/{dataset}.yaml if needed.
DEFAULT_BOUNDS = {
    'lr':         (3e-5, 3e-3),
    'kl_weight':  (0.05, 0.5),
    'lw_cont':    (3e-3, 0.1),
    'lw_label':   (0.05, 1.0),
}


def suggest_params(trial, bounds=None):
    """Ask Optuna for one HP point under log-uniform priors."""
    b = dict(DEFAULT_BOUNDS)
    if bounds:
        b.update(bounds)
    return {
        'lr':        trial.suggest_float('lr',        *b['lr'],        log=True),
        'kl_weight': trial.suggest_float('kl_weight', *b['kl_weight'], log=True),
        'lw_cont':   trial.suggest_float('lw_cont',   *b['lw_cont'],   log=True),
        'lw_label':  trial.suggest_float('lw_label',  *b['lw_label'],  log=True),
    }


def loss_weights_string(params):
    """BiGG -loss_weights takes a "cont,label" comma string."""
    return f"{params['lw_cont']},{params['lw_label']}"
