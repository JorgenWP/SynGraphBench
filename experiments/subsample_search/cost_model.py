"""Cost model for BiGG subsample training + generation wall-clock.

Fits on `experiments/bigg_capacity/results.csv` (status=ok rows). Predicts
per-subgraph training and generation cost for any `(dataset, avg_edges)` pair
using two pooled OLS regressions in log space with per-dataset intercepts:

    log(t_train_per_sg_per_ep) = slope_train * log(m) + intercept_train[dataset]
    log(t_gen_per_sg)          = slope_gen   * log(m) + intercept_gen[dataset]

`avg_nodes` is omitted (collinear with `avg_edges` and BiGG cost is per-edge).

Used by `plan_grid.py` to set K per cell from a 1-hour / 300-epoch budget."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAPACITY_CSV = os.path.normpath(os.path.join(_HERE, '..', 'bigg_capacity', 'results.csv'))
_CAPACITY_NUM_EPOCHS = 5  # capacity_benchmark.py --epochs 5 default (line 374)


@dataclass(frozen=True)
class CostModel:
    datasets: tuple[str, ...]
    slope_train: float
    intercept_train: dict[str, float]
    sigma_train: float
    slope_gen: float
    intercept_gen: dict[str, float]
    sigma_gen: float
    slope_vram: float
    intercept_vram: dict[str, float]
    sigma_vram: float
    m_ceiling: dict[str, float]

    def predict_peak_vram_gb(
        self,
        dataset: str,
        m: float,
        conf_sigma: float = 0.5,
    ) -> Optional[float]:
        """Return predicted peak VRAM in GB for one subgraph of avg_edges=m.

        Peak VRAM is per-subgraph (training processes one subgraph at a time on
        GPU), so K doesn't factor in. `conf_sigma=0.5` matches the wall-clock
        buffer — ~70th percentile of the log-normal residual. Fit is on
        `peak_vram_mb`; we divide by 1024 here so callers work in GB."""
        if m <= 0 or dataset not in self.intercept_vram:
            return None
        log_v = (self.intercept_vram[dataset]
                 + self.slope_vram * math.log(m)
                 + conf_sigma * self.sigma_vram)
        return math.exp(log_v) / 1024.0

    def predict_per_sg_total(
        self,
        dataset: str,
        m: float,
        target_full_epochs: int = 300,
        conf_sigma: float = 0.5,
    ) -> Optional[float]:
        """Return predicted per-subgraph wall-clock (train+gen) in seconds.

        `conf_sigma=0.5` adds ~0.5σ in log space — a conservative buffer so the
        prediction sits at ~70th percentile of the log-normal residual."""
        if m <= 0 or dataset not in self.intercept_train:
            return None
        log_m = math.log(m)
        log_t_train = (self.intercept_train[dataset]
                       + self.slope_train * log_m
                       + conf_sigma * self.sigma_train)
        log_t_gen = (self.intercept_gen[dataset]
                     + self.slope_gen * log_m
                     + conf_sigma * self.sigma_gen)
        return target_full_epochs * math.exp(log_t_train) + math.exp(log_t_gen)

    def predict_K(
        self,
        dataset: str,
        m: float,
        budget_s: float = 3600.0,
        target_full_epochs: int = 300,
        conf_sigma: float = 0.5,
    ) -> Optional[int]:
        per_sg = self.predict_per_sg_total(dataset, m, target_full_epochs, conf_sigma)
        if per_sg is None or per_sg <= 0:
            return None
        return int(budget_s // per_sg)


def load_capacity_rows(csv_path: str = _CAPACITY_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['status'] == 'ok'].copy()
    for c in ['num_train_subgraphs', 'num_gen_subgraphs', 'train_seconds',
              'gen_seconds', 'avg_nodes', 'avg_edges', 'peak_vram_mb']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    df['t_train_per_sg_per_ep'] = (
        df['train_seconds'] / (_CAPACITY_NUM_EPOCHS * df['num_train_subgraphs'])
    )
    df['t_gen_per_sg'] = df['gen_seconds'] / df['num_gen_subgraphs']
    return df


def _fit_one(df: pd.DataFrame, y_col: str, datasets: tuple[str, ...]):
    """Fit log(y) = slope * log(avg_edges) + sum_d alpha_d * I(dataset=d).

    No global intercept — each dataset gets its own intercept directly.
    Returns (slope, {dataset: intercept}, residual_sigma_in_log_space)."""
    n = len(df)
    log_y = np.log(df[y_col].to_numpy())
    log_m = np.log(df['avg_edges'].to_numpy())
    D = np.zeros((n, len(datasets)))
    ds_arr = df['dataset'].to_numpy()
    for j, d in enumerate(datasets):
        D[:, j] = (ds_arr == d).astype(float)
    X = np.column_stack([log_m, D])
    coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    slope = float(coef[0])
    intercepts = {d: float(coef[1 + j]) for j, d in enumerate(datasets)}
    preds = X @ coef
    sigma = float(np.sqrt(((log_y - preds) ** 2).sum() / max(1, n - len(coef))))
    return slope, intercepts, sigma


def fit_cost_model(csv_path: str = _CAPACITY_CSV) -> CostModel:
    df = load_capacity_rows(csv_path)
    datasets = tuple(sorted(df['dataset'].unique()))
    slope_t, int_t, sig_t = _fit_one(df, 't_train_per_sg_per_ep', datasets)
    slope_g, int_g, sig_g = _fit_one(df, 't_gen_per_sg', datasets)
    slope_v, int_v, sig_v = _fit_one(df, 'peak_vram_mb', datasets)
    m_ceiling = df.groupby('dataset')['avg_edges'].max().to_dict()
    return CostModel(
        datasets=datasets,
        slope_train=slope_t, intercept_train=int_t, sigma_train=sig_t,
        slope_gen=slope_g, intercept_gen=int_g, sigma_gen=sig_g,
        slope_vram=slope_v, intercept_vram=int_v, sigma_vram=sig_v,
        m_ceiling=m_ceiling,
    )


def dataset_m_ceiling(model: CostModel, dataset: str) -> float:
    """Largest observed avg_edges among ok rows for `dataset`.

    Returns +inf for unknown datasets (caller decides whether to skip)."""
    return model.m_ceiling.get(dataset, float('inf'))


def _self_check(model: CostModel, df: pd.DataFrame) -> None:
    rows = []
    for _, r in df.iterrows():
        observed_raw = str(r['extrap_K_at_300ep']).strip()
        if observed_raw == '' or observed_raw == 'nan':
            continue
        observed = int(float(observed_raw))
        pred_buf = model.predict_K(r['dataset'], float(r['avg_edges']), conf_sigma=0.5)
        pred_raw = model.predict_K(r['dataset'], float(r['avg_edges']), conf_sigma=0.0)
        rows.append({
            'dataset': r['dataset'],
            'method_tag': r['method_tag'],
            'avg_edges': float(r['avg_edges']),
            'observed_K': observed,
            'pred_K_raw': pred_raw,
            'pred_K_buf': pred_buf,
            'ratio_raw': pred_raw / observed if observed else float('nan'),
            'ratio_buf': pred_buf / observed if observed else float('nan'),
        })
    out = pd.DataFrame(rows)
    print('\n--- Self-consistency: predicted_K vs CSV extrap_K_at_300ep ---')
    print(out.to_string(index=False))
    for col in ['ratio_raw', 'ratio_buf']:
        vals = out[col].dropna()
        print(f'\n{col}:  median={vals.median():.2f}  p25={vals.quantile(0.25):.2f}  '
              f'p75={vals.quantile(0.75):.2f}  '
              f'frac within ±30% of 1.0 = {((vals >= 0.7) & (vals <= 1.3)).mean():.1%}')


def main() -> None:
    df = load_capacity_rows()
    print(f'Capacity ok rows: {len(df)}')
    print(f'Datasets: {sorted(df["dataset"].unique())}')
    model = fit_cost_model()
    print('\n--- Fitted cost model ---')
    print(f'train  log(t_per_sg_per_ep)  =  {model.slope_train:+.3f} * log(m)  +  dataset_intercept')
    print(f'       residual sigma (log)  =  {model.sigma_train:.3f}')
    print(f'gen    log(t_per_sg)         =  {model.slope_gen:+.3f} * log(m)  +  dataset_intercept')
    print(f'       residual sigma (log)  =  {model.sigma_gen:.3f}')
    print(f'vram   log(peak_vram_mb)     =  {model.slope_vram:+.3f} * log(m)  +  dataset_intercept')
    print(f'       residual sigma (log)  =  {model.sigma_vram:.3f}')
    print('\n--- Per-dataset intercepts (log seconds / log MB) and m_ceiling ---')
    for d in model.datasets:
        print(f'  {d:<10}  train_int={model.intercept_train[d]:+.3f}  '
              f'gen_int={model.intercept_gen[d]:+.3f}  '
              f'vram_int={model.intercept_vram[d]:+.3f}  '
              f'm_ceiling={model.m_ceiling[d]:>10,.0f}')
    _self_check(model, df)


if __name__ == '__main__':
    main()
