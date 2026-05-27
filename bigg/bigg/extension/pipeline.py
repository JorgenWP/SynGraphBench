import os
import sys
import json
import time
import pickle
import traceback
import argparse
import psutil
import dgl
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm

# BiGG specific imports
from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.experiments.train_utils import sqrtn_forward_backward

# Import both custom models
from bigg.extension.customized_models import (
    BiggWithConditionedFeats,
    BiggWithFeatsAndLabels,
    BiggWithARCatFeats,
    BiggWithMDNFeats,
)

# Preprocessing utilities
from bigg.extension.preprocessing import (
    load_dgl_graph,
    dgl_to_networkx,
    bfs_reorder,
    normalize_features,
    detect_binary_columns,
    build_generated_dgl,
    partition_graph_bfs,
    forest_fire_subsample,
    fit_feature_bins,
    default_bin_sigma,
    NORMALIZATION_METHODS,
)

def compute_kl_beta(epoch, schedule, anneal_epochs=None,
                    cycle_epochs=None, ramp_ratio=0.5):
    """Return β ∈ [0, 1] for the current epoch.

    Effective KL coefficient = β × kl_weight (kl_weight is the user-set ceiling).

    Schedules:
      - 'none':    β ≡ 1.0 (constant; preserves pre-annealing behavior).
      - 'linear':  β ramps 0 → 1 over `anneal_epochs`, stays at 1 thereafter.
      - 'cyclic':  Fu et al. 2019 — β resets to 0 every `cycle_epochs`, ramps
                   linearly to 1 over the first `ramp_ratio` fraction of the
                   cycle, then sits at 1 for the remainder.
    """
    if schedule == 'none':
        return 1.0
    if schedule == 'linear':
        return min(1.0, epoch / max(anneal_epochs, 1))
    if schedule == 'cyclic':
        cycle_pos = (epoch % max(cycle_epochs, 1)) / max(cycle_epochs, 1)
        if ramp_ratio <= 0 or cycle_pos >= ramp_ratio:
            return 1.0
        return cycle_pos / ramp_ratio
    raise ValueError(f"unknown KL schedule: {schedule!r}")


def main():

    #Parse pipeline args
    pipeline_parser = argparse.ArgumentParser(allow_abbrev=False)
    pipeline_parser.add_argument('-model_type', type=str, default='conditional',
                                 choices=['conditional', 'independent'],
                                 help='Choose feature generation model')
    pipeline_parser.add_argument('-label_temp', type=float, default=1.0,
                                 help='Sampling temperature for label generation (>1 boosts minority classes)')
    pipeline_parser.add_argument('-noise_std', type=float, default=0.0,
                                 help='Gaussian noise std added to hidden state during training')
    pipeline_parser.add_argument('-ss_max_prob', type=float, default=0.0,
                                 help='Max scheduled sampling probability (0 = disabled)')
    pipeline_parser.add_argument('-ss_start_epoch', type=int, default=0,
                                 help='Epoch to begin scheduled sampling annealing')
    pipeline_parser.add_argument('-bfs_preprocess', type=eval, default=False,
                                 help='Apply fixed BFS node ordering before training')
    pipeline_parser.add_argument('-normalize', type=str, default=None,
                                 choices=list(NORMALIZATION_METHODS),
                                 help='Feature normalisation method (default: none)')
    pipeline_parser.add_argument('-loss_weights', type=str, default='1,1',
                                 help='Loss weights for cont,label relative to struct (default: 1,1). '
                                      'Applied on top of dynamic normalization after epoch 0.')
    pipeline_parser.add_argument('--hetero_feat', action='store_true', default=False,
                                 help='Heteroscedastic feature prediction: predict mean + log-variance '
                                      'and sample at generation time (default: deterministic MSE)')
    pipeline_parser.add_argument('-logvar_floor', type=float, default=-4.0,
                                 help='Lower clamp for log-variance in hetero_feat mode. '
                                      'Lower values allow tighter distributions (default: -4.0)')
    pipeline_parser.add_argument('--binary_feat', action='store_true', default=False,
                                 help='Auto-detect binary feature columns and use BCE loss + '
                                      'Bernoulli sampling instead of Gaussian head (default: off)')
    pipeline_parser.add_argument('--vae_feat', action='store_true', default=False,
                                 help='Add a per-node label-agnostic CVAE latent shared across feature '
                                      'decoders to capture cross-feature covariance (default: off)')
    pipeline_parser.add_argument('-vae_dim', type=int, default=16,
                                 help='Latent dimensionality when --vae_feat is on (default: 16)')
    pipeline_parser.add_argument('-kl_weight', type=float, default=1.0,
                                 help='Coefficient on the KL term in the VAE ELBO (default: 1.0). '
                                      'Not subject to dynamic calibration.')
    pipeline_parser.add_argument('--cat_feat', action='store_true', default=False,
                                 help='Use autoregressive categorical feature predictor: per-feature '
                                      'quantile bins, value-space Gaussian soft-label cross-entropy. '
                                      'Mutually exclusive with --hetero_feat and --vae_feat.')
    pipeline_parser.add_argument('-n_bins', type=int, default=32,
                                 help='Number of quantile bins per continuous feature when --cat_feat '
                                      'is on (default: 32)')
    pipeline_parser.add_argument('-bin_sigma', type=float, default=None,
                                 help='Gaussian soft-label std in feature-value units when --cat_feat '
                                      'is on. Default: 0.5 x median bin-center spacing.')
    pipeline_parser.add_argument('--mdn_feat', action='store_true', default=False,
                                 help='Use Mixture Density Network feature head: per-feature '
                                      'K-component Gaussian mixture conditioned on [h, label_embed]. '
                                      'Mutually exclusive with --hetero_feat / --vae_feat / --cat_feat.')
    pipeline_parser.add_argument('-mdn_components', type=int, default=8,
                                 help='Number of mixture components per feature when --mdn_feat '
                                      'is on (default: 8)')
    pipeline_parser.add_argument('-mdn_logsigma_floor', type=float, default=-4.0,
                                 help='Lower clamp for MDN component log-sigma (default: -4.0)')
    pipeline_parser.add_argument('-mdn_base', type=str, default='gaussian',
                                 choices=['gaussian', 'logit_normal'],
                                 help='Per-component base distribution for the MDN head '
                                      '(default: gaussian). Use logit_normal for [0,1] targets '
                                      '(pairs with --normalize cdf).')
    pipeline_parser.add_argument('--mask_test_labels', action='store_true', default=False,
                                 help='Exclude test node labels (split 0) from label loss to prevent '
                                      'data leakage in anomaly detection benchmarks')
    pipeline_parser.add_argument('--save_model', action='store_true', default=False,
                                 help='Persist model.state_dict + reconstruction args to model.pt '
                                      'alongside the synthetic outputs after training, for offline '
                                      'diagnostics (default: off).')
    pipeline_parser.add_argument('-save_model_every', type=int, default=0,
                                 help='If > 0 (and --save_model is on), also persist a snapshot '
                                      'model_epoch{N}.pt every N epochs during training. '
                                      'Final model.pt is still written at end. Default: 0 (off).')
    pipeline_parser.add_argument('--subsample', action='store_true', default=False,
                                 help='Sample subgraphs via forest fire for VRAM-limited training')
    pipeline_parser.add_argument('-subsample_size', type=int, default=2000,
                                 help='Target number of nodes per subgraph (default: 2000)')
    pipeline_parser.add_argument('-burn_prob', type=float, default=0.3,
                                 help='Forest fire burn probability — controls subgraph density (default: 0.3)')
    pipeline_parser.add_argument('-num_subgraphs', type=int, default=None,
                                 help='Number of subgraphs to generate. Default: ceil(N / subsample_size)')
    pipeline_parser.add_argument('--load_subsamples', action='store_true', default=False,
                                 help='Load pre-computed training subgraphs from '
                                      'datasets/bigg_subsamples/<dataset>/<subsampling_config>/'
                                      'split<split_id>.pkl instead of sampling at runtime. '
                                      'Mutually exclusive with --subsample.')
    pipeline_parser.add_argument('-subsampling_config', type=str, default=None,
                                 help='params_tag of the pre-computed subsample set to load '
                                      "(e.g. 'ff_b0.5_M1', 'metis_K5'). Required when "
                                      '--load_subsamples is set.')
    pipeline_parser.add_argument('-split_id', type=int, default=0,
                                 help='GADBench split id (0..4) selecting which '
                                      'split<id>.pkl to load (default: 0).')
    pipeline_parser.add_argument('-min_subgraph_nodes', type=int, default=0,
                                 help='Drop loaded partitions with fewer than N '
                                      'nodes. Applied after pickle load under '
                                      '--load_subsamples; default 0 = no filter.')
    pipeline_parser.add_argument('-kl_schedule', type=str, default='none',
                                 choices=['none', 'linear', 'cyclic'],
                                 help='KL annealing schedule for the VAE coefficient. '
                                      '"linear" ramps β from 0 to kl_weight over -kl_anneal_epochs. '
                                      '"cyclic" (Fu et al. NAACL 2019) resets β every -kl_cycle_epochs '
                                      'and ramps over the first -kl_ramp_ratio fraction of the cycle. '
                                      'Use to address posterior collapse (default: none).')
    pipeline_parser.add_argument('-kl_anneal_epochs', type=int, default=0,
                                 help='Epochs over which β linearly ramps 0 → kl_weight when '
                                      '-kl_schedule linear (required for linear).')
    pipeline_parser.add_argument('-kl_cycle_epochs', type=int, default=0,
                                 help='Epochs per cycle when -kl_schedule cyclic (required for cyclic). '
                                      'Paper M = ceil(num_epochs / kl_cycle_epochs).')
    pipeline_parser.add_argument('-kl_ramp_ratio', type=float, default=0.5,
                                 help='Fraction of cycle spent ramping when -kl_schedule cyclic '
                                      '(paper R, default 0.5). 0.5 → first 50%% ramps 0→1, last 50%% '
                                      'sits at 1.')
    pipeline_parser.add_argument('-recal_momentum', type=float, default=1.0,
                                 help='EMA momentum for dynamic loss-weight recalibration. '
                                      'Each epoch updates ema = m*ema + (1-m)*current per component, '
                                      'then recomputes w_cont/w_bin/w_label from the EMA ratios '
                                      '(user_w_cont/user_w_label from -loss_weights still apply as '
                                      'multiplicative priors). Range [0,1]. 1.0 (default) disables '
                                      'recalibration — weights stay at the one-time calibration '
                                      'values, bit-identical to pre-feature behavior. 0.0 = no '
                                      'smoothing (track latest epoch exactly). Typical: 0.9 (~10 '
                                      'epoch horizon), 0.99 (~100 epoch horizon). KL is unaffected '
                                      '(annealed separately via -kl_schedule).')
    pipeline_parser.add_argument('-subsample_method', type=str, default='forest_fire',
                                 choices=['forest_fire', 'metis'],
                                 help='Subsampling method when --subsample is on. "forest_fire" '
                                      'uses bigg\'s primitive (default; preserves existing behavior). '
                                      '"metis" partitions the graph into K disjoint parts via pymetis.')
    pipeline_parser.add_argument('-multiplicity_cap', type=str, default='minf',
                                 choices=['m1', 'm2', 'minf'],
                                 help='Per-node multiplicity cap across forest_fire draws. '
                                      '"m1" = disjoint, "m2" = up to 2 draws per node, '
                                      '"minf" = no cap (default; preserves existing behavior). '
                                      'Ignored for metis (M=1 by construction).')
    pipeline_parser.add_argument('-num_train_subgraphs', type=int, default=None,
                                 help='Cap the training inner loop at the first N partitions '
                                      '(default: all sampled subgraphs). For capacity benchmarking.')
    pipeline_parser.add_argument('-num_gen_subgraphs', type=int, default=None,
                                 help='Cap the generation loop at the first N partitions '
                                      '(default: same as num_train_subgraphs). For capacity benchmarking.')
    pipeline_parser.add_argument('-timing_log_path', type=str, default=None,
                                 help='If set, write a JSON timing log (train/gen seconds, peak VRAM, '
                                      'partition stats, status) to this path. Always written even on '
                                      'failure; pipeline exits with code 2 on caught error.')
    pipeline_parser.add_argument('-input_graph', type=str, default=None,
                                 help='Override path to input DGL graph (e.g. a k-anonymized variant '
                                      'at datasets/kanon/<dataset>/<stem>.dgl). When set, supersedes '
                                      'the default ../datasets/original/<dataset> lookup. -dataset '
                                      'is still used for save naming.')

    pipeline_args, _ = pipeline_parser.parse_known_args()

    # Parse loss weights
    lw = [float(x) for x in pipeline_args.loss_weights.split(',')]
    assert len(lw) == 2, f'Expected 2 loss weights (cont,label), got {len(lw)}'
    user_w_cont, user_w_label = lw

    if not (0.0 <= pipeline_args.recal_momentum <= 1.0):
        raise ValueError(f'-recal_momentum must be in [0, 1], got {pipeline_args.recal_momentum}')

    # AR categorical is its own predictor; reject combinations that don't apply.
    if pipeline_args.cat_feat and (pipeline_args.hetero_feat or pipeline_args.vae_feat):
        raise ValueError('--cat_feat is mutually exclusive with --hetero_feat and --vae_feat')

    # MDN replaces the continuous head entirely; reject combinations that don't apply.
    # MDN composes with VAE (z is concatenated into the head's conditioning input).
    if pipeline_args.mdn_feat and (pipeline_args.hetero_feat or pipeline_args.cat_feat):
        raise ValueError('--mdn_feat is mutually exclusive with --hetero_feat, --cat_feat')

    # CDF encoding bounds targets to [0,1]. Compatible with the default sigmoid+BCE
    # head and with --mdn_feat when --mdn_base logit_normal (components live on the
    # logit and are mapped to (0,1) via sigmoid at sample time). All other cont-head
    # modes assume unbounded targets and are incompatible.
    if pipeline_args.normalize == 'cdf':
        bad = []
        if pipeline_args.hetero_feat:
            bad.append('--hetero_feat')
        if pipeline_args.cat_feat:
            bad.append('--cat_feat')
        if pipeline_args.mdn_feat and pipeline_args.mdn_base != 'logit_normal':
            bad.append('--mdn_feat (without --mdn_base logit_normal)')
        if bad:
            raise ValueError('--normalize cdf is mutually exclusive with: ' + ', '.join(bad))

    # KL annealing schedule validation
    if pipeline_args.kl_schedule == 'linear' and pipeline_args.kl_anneal_epochs <= 0:
        raise ValueError('-kl_schedule linear requires -kl_anneal_epochs > 0')
    if pipeline_args.kl_schedule == 'cyclic':
        if pipeline_args.kl_cycle_epochs <= 0:
            raise ValueError('-kl_schedule cyclic requires -kl_cycle_epochs > 0')
        if not (0 < pipeline_args.kl_ramp_ratio <= 1):
            raise ValueError('-kl_ramp_ratio must satisfy 0 < ratio <= 1')
    if pipeline_args.kl_schedule != 'none' and not pipeline_args.vae_feat:
        print(f'WARNING: -kl_schedule={pipeline_args.kl_schedule} has no effect '
              'when --vae_feat is off (kl_weight is unused).')
    if (pipeline_args.kl_schedule == 'linear'
            and pipeline_args.kl_anneal_epochs > cmd_args.num_epochs):
        print(f'WARNING: -kl_anneal_epochs ({pipeline_args.kl_anneal_epochs}) > num_epochs '
              f'({cmd_args.num_epochs}); β will never reach 1.')
    if (pipeline_args.kl_schedule == 'cyclic'
            and pipeline_args.kl_cycle_epochs > cmd_args.num_epochs):
        print(f'WARNING: -kl_cycle_epochs ({pipeline_args.kl_cycle_epochs}) > num_epochs '
              f'({cmd_args.num_epochs}); β will never reach 1.')

    # --load_subsamples disables runtime sampling; the two modes are exclusive.
    if pipeline_args.load_subsamples and pipeline_args.subsample:
        raise ValueError('--load_subsamples and --subsample are mutually exclusive')
    if pipeline_args.load_subsamples and not pipeline_args.subsampling_config:
        raise ValueError('--load_subsamples requires -subsampling_config <params_tag>')
    if pipeline_args.load_subsamples:
        # Warn on runtime-sampling knobs that have no effect in load mode.
        _ignored = []
        if pipeline_args.subsample_size != 2000:
            _ignored.append(f'-subsample_size={pipeline_args.subsample_size}')
        if pipeline_args.burn_prob != 0.3:
            _ignored.append(f'-burn_prob={pipeline_args.burn_prob}')
        if pipeline_args.num_subgraphs is not None:
            _ignored.append(f'-num_subgraphs={pipeline_args.num_subgraphs}')
        if pipeline_args.subsample_method != 'forest_fire':
            _ignored.append(f'-subsample_method={pipeline_args.subsample_method}')
        if pipeline_args.multiplicity_cap != 'minf':
            _ignored.append(f'-multiplicity_cap={pipeline_args.multiplicity_cap}')
        if _ignored:
            print('WARNING: --load_subsamples ignores runtime-sampling args: '
                  + ', '.join(_ignored))

    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)

    #Load dataset
    DATASET = cmd_args.data_dir
    graph = load_dgl_graph(DATASET, input_path=pipeline_args.input_graph)

    #Get features and labels
    cont_feats = graph.ndata['feature']
    labels = graph.ndata['label']

    # Detect binary columns before normalization (binary features skip normalization)
    binary_idx = []
    if pipeline_args.binary_feat:
        binary_idx = detect_binary_columns(cont_feats)
        print(f'Binary feature columns detected: {binary_idx} '
              f'({len(binary_idx)} binary, {cont_feats.shape[1] - len(binary_idx)} continuous)')

    # Optionally normalise features (only non-binary columns)
    norm_stats = None
    if pipeline_args.normalize is not None:
        if binary_idx:
            cont_idx = sorted(set(range(cont_feats.shape[1])) - set(binary_idx))
            normed, norm_stats = normalize_features(cont_feats[:, cont_idx], pipeline_args.normalize)
            cont_feats = cont_feats.clone()
            cont_feats[:, cont_idx] = normed
            norm_stats['binary_idx'] = binary_idx
        else:
            cont_feats, norm_stats = normalize_features(cont_feats, pipeline_args.normalize)
        print(f'Applied {pipeline_args.normalize} normalisation to features')

    #Determine dimensions of features
    feat_dim = cont_feats.shape[1]
    num_classes = int(labels.max().item()) + 1

    node_data = torch.cat([cont_feats, labels.unsqueeze(1).float()], dim = 1).to(cmd_args.device)
    list_node_feats = [node_data]

    # Build label mask: True = include in label loss (train+val nodes from split 0)
    label_mask = None
    if pipeline_args.mask_test_labels:
        train_m = graph.ndata['train_masks'][:, 0].bool()
        val_m = graph.ndata['val_masks'][:, 0].bool()
        label_mask = (train_m | val_m).to(cmd_args.device)
        n_masked = (~label_mask).sum().item()
        print(f'Label masking enabled: {n_masked} test nodes excluded from label loss (split 0)')

    #Convert topology for treelib
    graph_nx = dgl_to_networkx(graph)

    # --- Capacity-benchmark timing state ---
    # Always populated; only flushed to disk when -timing_log_path is set.
    # Phases (sampling/training/generation) update this in-place and the
    # finalizer writes JSON regardless of success/failure.
    timing_state = {
        'dataset': DATASET,
        'subsample_method': pipeline_args.subsample_method,
        'multiplicity_cap': pipeline_args.multiplicity_cap,
        'num_train_subgraphs': None,
        'num_gen_subgraphs': None,
        'num_epochs': cmd_args.num_epochs,
        'partitions': [],
        'avg_nodes': None,
        'avg_edges': None,
        'avg_density': None,
        'train_seconds': None,
        'gen_seconds': None,
        'peak_vram_mb': None,
        'peak_ram_mb': None,
        'status': 'ok',
        'error_msg': None,
        'failure_phase': None,
    }

    def _write_timing_log():
        if pipeline_args.timing_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(pipeline_args.timing_log_path)),
                        exist_ok=True)
            with open(pipeline_args.timing_log_path, 'w') as f:
                json.dump(timing_state, f, indent=2)

    def _record_failure(phase, exc):
        msg = str(exc)
        is_oom = (isinstance(exc, torch.cuda.OutOfMemoryError)
                  or 'out of memory' in msg.lower())
        timing_state['failure_phase'] = phase
        timing_state['status'] = f'oom_{phase}' if is_oom else 'other_error'
        timing_state['error_msg'] = traceback.format_exc()[-2000:]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # --- Build list of (gid, sub_node_data, sub_label_mask, sub_num_nodes) ---
    # When subsampling is disabled this list has a single entry for the full graph.
    try:
        raw_partitions = None  # set by --subsample or --load_subsamples branches

        if pipeline_args.load_subsamples:
            # Lazy bootstrap of experiments/subsample_search/splitsource. The
            # persisted pickle's orig_idx values index into the split's
            # train+val LCC-relabeled space, so we need the SplitSource to
            # map them back to full-graph node ids.
            _here = os.path.dirname(os.path.abspath(__file__))
            _proj_root = os.path.abspath(os.path.join(_here, '..', '..', '..'))
            _sampler_dir = os.path.join(_proj_root, 'experiments', 'subsample_search')
            if _sampler_dir not in sys.path:
                sys.path.insert(0, _sampler_dir)
            from splitsource import load_split_source  # noqa: E402

            data_dir_abs = os.path.join(_proj_root, 'datasets', 'original')
            print(f'Loading subsamples from disk: config={pipeline_args.subsampling_config}, '
                  f'split_id={pipeline_args.split_id}')
            src = load_split_source(DATASET, pipeline_args.split_id, data_dir=data_dir_abs)

            pkl_path = os.path.join(_proj_root, 'datasets', 'bigg_subsamples', DATASET,
                                    pipeline_args.subsampling_config,
                                    f'split{pipeline_args.split_id}.pkl')
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(
                    f'Subsample pickle not found: {pkl_path}. '
                    f'Generate it via experiments/subsample_search/run_grid.py first.')
            with open(pkl_path, 'rb') as f:
                payload = pickle.load(f)
            print(f'  Loaded {len(payload["partitions"])} partitions (meta={payload["meta"]})')

            # The pickled sub_nd is unnormalized LCC-space features+label; we
            # discard it and re-extract from the (already-normalized) full
            # graph node_data so the normalization population matches the
            # runtime --subsample path.
            orig_ids_full = src.orig_ids.long()
            raw_partitions = []
            for sg_nx, _stored_sub_nd, orig_idx_lcc in payload['partitions']:
                full_idx = orig_ids_full[orig_idx_lcc.long()]
                sub_nd = node_data[full_idx.to(node_data.device)]
                # Saved subgraphs come from dgl.to_networkx().to_undirected(),
                # which yields a MultiGraph; TreeLib expects a simple Graph.
                if sg_nx.is_multigraph():
                    sg_simple = nx.Graph()
                    sg_simple.add_nodes_from(sg_nx.nodes())
                    sg_simple.add_edges_from(sg_nx.edges())
                    sg_nx = sg_simple
                raw_partitions.append((sg_nx, sub_nd, full_idx))
            if pipeline_args.min_subgraph_nodes > 0:
                kept, dropped_sizes = [], []
                for sg, nd, fi in raw_partitions:
                    if sg.number_of_nodes() >= pipeline_args.min_subgraph_nodes:
                        kept.append((sg, nd, fi))
                    else:
                        dropped_sizes.append(sg.number_of_nodes())
                if dropped_sizes:
                    print(f'  Filtered {len(dropped_sizes)} partitions below '
                          f'{pipeline_args.min_subgraph_nodes} nodes (sizes: {dropped_sizes})')
                raw_partitions = kept
            print(f'  Subgraph sizes: {[sg.number_of_nodes() for sg, _, _ in raw_partitions]}')

        elif pipeline_args.subsample:
            import math
            n_total = graph_nx.number_of_nodes()
            num_subgraphs = pipeline_args.num_subgraphs or math.ceil(n_total / pipeline_args.subsample_size)

            # Dispatch: keep the legacy bigg primitive for the default
            # (forest_fire + minf) path so existing scripts are bit-identical.
            # Otherwise route through the richer sampler in
            # experiments/subsample_search/sampling.py.
            use_legacy = (pipeline_args.subsample_method == 'forest_fire'
                          and pipeline_args.multiplicity_cap == 'minf')

            if use_legacy:
                print(f'Subsampling enabled: {num_subgraphs} forest fire samples, '
                      f'target_size={pipeline_args.subsample_size}, burn_prob={pipeline_args.burn_prob}')
                raw_partitions = []
                for _ in range(num_subgraphs):
                    raw_partitions.append(forest_fire_subsample(
                        graph_nx, node_data,
                        target_size=pipeline_args.subsample_size,
                        burn_prob=pipeline_args.burn_prob,
                    ))
            else:
                # Lazy bootstrap of experiments/subsample_search/sampling
                _here = os.path.dirname(os.path.abspath(__file__))
                _proj_root = os.path.abspath(os.path.join(_here, '..', '..', '..'))
                _sampler_dir = os.path.join(_proj_root, 'experiments', 'subsample_search')
                if _sampler_dir not in sys.path:
                    sys.path.insert(0, _sampler_dir)
                from sampling import sample_with_method  # noqa: E402

                cap_map = {'m1': 1, 'm2': 2, 'minf': None}
                mcap = cap_map[pipeline_args.multiplicity_cap]
                print(f'Subsampling enabled (method={pipeline_args.subsample_method}, '
                      f'multiplicity_cap={pipeline_args.multiplicity_cap}): '
                      f'K={num_subgraphs}, target_size={pipeline_args.subsample_size}, '
                      f'burn_prob={pipeline_args.burn_prob}')
                raw_partitions = sample_with_method(
                    method=pipeline_args.subsample_method,
                    g_nx=graph_nx,
                    node_data=node_data,
                    K=num_subgraphs,
                    seed=cmd_args.seed,
                    multiplicity_cap=(1 if pipeline_args.subsample_method == 'metis' else mcap),
                    target_size=pipeline_args.subsample_size,
                    burn_prob=pipeline_args.burn_prob,
                )

            print(f'  Produced {len(raw_partitions)} samples '
                  f'(sizes: {[sg.number_of_nodes() for sg, _, _ in raw_partitions]})')

        if raw_partitions is not None:
            # Optionally cap the trained-on set (capacity benchmark trains on N of K).
            if pipeline_args.num_train_subgraphs is not None:
                raw_partitions = raw_partitions[:pipeline_args.num_train_subgraphs]
                print(f'  Capped to first {len(raw_partitions)} partitions for training '
                      f'(num_train_subgraphs={pipeline_args.num_train_subgraphs}).')

            subgraphs = []  # list of (gid, sub_node_data, sub_label_mask, sub_num_nodes)
            for sg, sub_nd, orig_idx in raw_partitions:
                if pipeline_args.bfs_preprocess:
                    sg, sub_nd, perm = bfs_reorder(sg, sub_nd)
                    sub_lm = label_mask[orig_idx[perm]] if label_mask is not None else None
                else:
                    sub_lm = label_mask[orig_idx] if label_mask is not None else None
                gid = TreeLib.InsertGraph(sg)
                subgraphs.append((gid, sub_nd.to(cmd_args.device), sub_lm, sg.number_of_nodes()))

            # Record per-partition (n, m, density) for the trained-on set.
            for sg_nx, _, _ in raw_partitions:
                n = sg_nx.number_of_nodes()
                m = sg_nx.number_of_edges()
                density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
                timing_state['partitions'].append({'n': n, 'm': m, 'density': density})
        else:
            # Single full graph
            if pipeline_args.bfs_preprocess:
                graph_nx, node_data, perm = bfs_reorder(graph_nx, node_data)
                list_node_feats = [node_data]
                if label_mask is not None:
                    label_mask = label_mask[perm]
            TreeLib.InsertGraph(graph_nx)
            subgraphs = [(0, node_data, label_mask, graph_nx.number_of_nodes())]
            n = graph_nx.number_of_nodes()
            m = graph_nx.number_of_edges()
            density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
            timing_state['partitions'].append({'n': n, 'm': m, 'density': density})
    except Exception as exc:
        _record_failure('sampling', exc)
        if pipeline_args.timing_log_path:
            _write_timing_log()
            print(f'[capacity] sampling phase failed ({timing_state["status"]}); '
                  f'JSON written to {pipeline_args.timing_log_path}.', file=sys.stderr)
            sys.exit(2)
        raise

    timing_state['num_train_subgraphs'] = len(subgraphs)
    if timing_state['partitions']:
        ns = [p['n'] for p in timing_state['partitions']]
        ms = [p['m'] for p in timing_state['partitions']]
        ds = [p['density'] for p in timing_state['partitions']]
        timing_state['avg_nodes'] = sum(ns) / len(ns)
        timing_state['avg_edges'] = sum(ms) / len(ms)
        timing_state['avg_density'] = sum(ds) / len(ds)

    cmd_args.has_node_feats = True
    cmd_args.max_num_nodes = max(sub_num_nodes for _, _, _, sub_num_nodes in subgraphs)

    # Fit per-feature quantile bins on continuous (non-binary) columns of the full
    # training feature matrix. Fit once on all subgraphs combined so every subgraph
    # sees the same binning (which also stores on the model for generation-time use).
    bin_edges = bin_centers = None
    if pipeline_args.cat_feat:
        cont_for_bins = cont_feats[:, [i for i in range(feat_dim) if i not in binary_idx]]
        bin_edges, bin_centers = fit_feature_bins(cont_for_bins, pipeline_args.n_bins)
        if pipeline_args.bin_sigma is None:
            # Per-feature sigma (F,) — data-adaptive across feature regimes.
            pipeline_args.bin_sigma = default_bin_sigma(bin_centers)
            sigma_summary = (f'per-feature '
                             f'[min={pipeline_args.bin_sigma.min():.4f}, '
                             f'median={pipeline_args.bin_sigma.median():.4f}, '
                             f'max={pipeline_args.bin_sigma.max():.4f}]')
        else:
            # Scalar override from CLI — broadcast to all features.
            pipeline_args.bin_sigma = torch.full(
                (bin_centers.shape[0],), float(pipeline_args.bin_sigma))
            sigma_summary = f'scalar={pipeline_args.bin_sigma[0]:.4f}'
        print(f'Fit {pipeline_args.n_bins} quantile bins on {cont_for_bins.shape[1]} continuous columns; '
              f'bin_sigma={sigma_summary}')

    if pipeline_args.cat_feat:
        model = BiggWithARCatFeats(cmd_args, feat_dim=feat_dim, num_classes=num_classes,
                                   n_bins=pipeline_args.n_bins,
                                   bin_edges=bin_edges,
                                   bin_centers=bin_centers,
                                   bin_sigma=pipeline_args.bin_sigma,
                                   label_temp=pipeline_args.label_temp,
                                   noise_std=pipeline_args.noise_std,
                                   binary_feat=pipeline_args.binary_feat,
                                   binary_idx=binary_idx).to(cmd_args.device)
    elif pipeline_args.mdn_feat:
        model = BiggWithMDNFeats(cmd_args, feat_dim=feat_dim, num_classes=num_classes,
                                 n_components=pipeline_args.mdn_components,
                                 label_temp=pipeline_args.label_temp,
                                 noise_std=pipeline_args.noise_std,
                                 logsigma_floor=pipeline_args.mdn_logsigma_floor,
                                 binary_feat=pipeline_args.binary_feat,
                                 binary_idx=binary_idx,
                                 vae_feat=pipeline_args.vae_feat,
                                 vae_dim=pipeline_args.vae_dim,
                                 kl_weight=pipeline_args.kl_weight,
                                 logvar_floor=pipeline_args.logvar_floor,
                                 mdn_base=pipeline_args.mdn_base).to(cmd_args.device)
    elif pipeline_args.model_type == 'conditional':
        model = BiggWithConditionedFeats(cmd_args, feat_dim=feat_dim, num_classes=num_classes,
                                         label_temp=pipeline_args.label_temp,
                                         noise_std=pipeline_args.noise_std,
                                         hetero_feat=pipeline_args.hetero_feat,
                                         logvar_floor=pipeline_args.logvar_floor,
                                         binary_feat=pipeline_args.binary_feat,
                                         binary_idx=binary_idx,
                                         vae_feat=pipeline_args.vae_feat,
                                         vae_dim=pipeline_args.vae_dim,
                                         kl_weight=pipeline_args.kl_weight,
                                         cdf_mode=(pipeline_args.normalize == 'cdf')).to(cmd_args.device)
    elif pipeline_args.model_type == 'independent':
        model = BiggWithFeatsAndLabels(cmd_args, feat_dim=feat_dim, num_classes=num_classes,
                                       label_temp=pipeline_args.label_temp,
                                       noise_std=pipeline_args.noise_std,
                                       hetero_feat=pipeline_args.hetero_feat,
                                       logvar_floor=pipeline_args.logvar_floor,
                                       binary_feat=pipeline_args.binary_feat,
                                       binary_idx=binary_idx,
                                       vae_feat=pipeline_args.vae_feat,
                                       vae_dim=pipeline_args.vae_dim,
                                       kl_weight=pipeline_args.kl_weight,
                                       cdf_mode=(pipeline_args.normalize == 'cdf')).to(cmd_args.device)

    # The user-set kl_weight is the schedule's β=1 ceiling; preserve it since
    # model.kl_weight will be mutated per-epoch by the annealing schedule.
    base_kl_weight = pipeline_args.kl_weight

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)

    # Parameter count breakdown — total + top-level children.
    def _count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'Model parameters — total: {_count_params(model):,}')
    for _name, _mod in model.named_children():
        _n = _count_params(_mod)
        if _n > 0:
            print(f'  {_name}: {_n:,}')

    # Grouping for per-epoch gradient norm logging. "feat_head" covers the
    # active feature decoder (MDN / AR-cat / Gaussian) so its signal is visible
    # whichever path we chose.
    feat_head_names = ['mdn_head', 'ar_head', 'nodefeat_pred', 'binfeat_pred',
                       'feat_pos_embed', 'prev_bin_embed']
    label_head_names = ['nodelabel_pred', 'nodelabel_encoding']
    struct_skip = set(feat_head_names) | set(label_head_names) | {
        'nodefeat_encoding', 'combiner', 'node_state_update', 'vae_encoder'}

    state_head_names = {'nodefeat_encoding', 'combiner', 'node_state_update', 'vae_encoder'}

    def _group_params():
        """Return {group: [param, ...]} for per-group gradient ops."""
        groups = {'feat': [], 'label': [], 'state': [], 'struct': []}
        for name, mod in model.named_children():
            params = list(mod.parameters())
            if not params:
                continue
            if name in feat_head_names:
                groups['feat'].extend(params)
            elif name in label_head_names:
                groups['label'].extend(params)
            elif name in state_head_names:
                groups['state'].extend(params)
            else:
                groups['struct'].extend(params)
        return groups

    def _group_grad_norms():
        import math as _m
        sums = {k: 0.0 for k in ('feat', 'label', 'state', 'struct')}
        for g_name, params in _group_params().items():
            for p in params:
                if p.grad is not None:
                    sums[g_name] += float((p.grad ** 2).sum().item())
        return {k: _m.sqrt(v) for k, v in sums.items()}

    model.train()

    ss_max_prob = pipeline_args.ss_max_prob
    ss_start_epoch = pipeline_args.ss_start_epoch
    ss_ramp_epochs = max(cmd_args.num_epochs - ss_start_epoch, 1)

    # Memory tracking
    process = psutil.Process(os.getpid())
    gpu_available = torch.cuda.is_available()
    peak_ram_mb = 0.0
    peak_vram_mb = 0.0
    if gpu_available:
        torch.cuda.reset_peak_memory_stats()

    def _capture_peak_mem():
        timing_state['peak_ram_mb'] = peak_ram_mb
        if gpu_available:
            try:
                cur_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
                timing_state['peak_vram_mb'] = max(peak_vram_mb, cur_peak)
            except Exception:
                timing_state['peak_vram_mb'] = peak_vram_mb

    # Build save name and directory (computed before training so periodic
    # checkpoint snapshots can write under the same out_dir / save_dir).
    norm_tag = pipeline_args.normalize if pipeline_args.normalize is not None else 'none'
    bfs_tag = 'bfs' if pipeline_args.bfs_preprocess else 'nobfs'
    lw_tag = pipeline_args.loss_weights.replace(',', '_')
    hetero_tag = 'hetero' if pipeline_args.hetero_feat else 'det'
    lvf_tag = f'_lvf{pipeline_args.logvar_floor}' if pipeline_args.hetero_feat else ''
    mask_tag = '_masked' if pipeline_args.mask_test_labels else ''
    bin_tag = '_binfeat' if pipeline_args.binary_feat else ''
    vae_tag = f'_vae{pipeline_args.vae_dim}_kl{pipeline_args.kl_weight}' if pipeline_args.vae_feat else ''
    if pipeline_args.vae_feat and pipeline_args.kl_schedule == 'linear':
        kl_sched_tag = f'_klan{pipeline_args.kl_anneal_epochs}'
    elif pipeline_args.vae_feat and pipeline_args.kl_schedule == 'cyclic':
        kl_sched_tag = f'_klcyc{pipeline_args.kl_cycle_epochs}r{int(pipeline_args.kl_ramp_ratio * 100)}'
    else:
        kl_sched_tag = ''
    cat_tag = f'_cat{pipeline_args.n_bins}_smed{pipeline_args.bin_sigma.median().item():.2f}' if pipeline_args.cat_feat else ''
    if pipeline_args.mdn_feat:
        mdn_base_tag = 'lnmdn' if pipeline_args.mdn_base == 'logit_normal' else 'mdn'
        mdn_tag = f'_{mdn_base_tag}{pipeline_args.mdn_components}_lsf{pipeline_args.mdn_logsigma_floor}'
    else:
        mdn_tag = ''
    recal_tag = f'_recalM{pipeline_args.recal_momentum}' if pipeline_args.recal_momentum < 1.0 else ''
    if pipeline_args.load_subsamples:
        sub_tag = (f'_loadsub_{pipeline_args.subsampling_config}'
                   f'_split{pipeline_args.split_id}_n{len(subgraphs)}')
    elif pipeline_args.subsample:
        sub_tag = f'_sub{len(subgraphs)}_size{pipeline_args.subsample_size}_p{pipeline_args.burn_prob}'
    else:
        sub_tag = ''
    save_name = f'blksize_{cmd_args.blksize}_b_{cmd_args.batch_size}_lr_{cmd_args.learning_rate}_epochs_{cmd_args.num_epochs}_noise_{pipeline_args.noise_std}_ss_{pipeline_args.ss_max_prob}_norm_{norm_tag}_{bfs_tag}_lw_{lw_tag}_{hetero_tag}{lvf_tag}{mask_tag}{bin_tag}{vae_tag}{kl_sched_tag}{cat_tag}{mdn_tag}{recal_tag}{sub_tag}'
    # BIGG_SYNTHETIC_SAVE_ROOT lets callers (e.g. the BO coordinator) redirect
    # synthetic outputs out of datasets/synthetic so BO scaffolding doesn't
    # clutter the canonical pipeline location. Unset → legacy behavior.
    _save_root_override = os.environ.get('BIGG_SYNTHETIC_SAVE_ROOT')
    save_dir = (_save_root_override if _save_root_override
                else f'../datasets/synthetic/bigg/{DATASET}/hidden_labels')
    use_subdir = pipeline_args.subsample or pipeline_args.load_subsamples
    os.makedirs(save_dir, exist_ok=True)
    if use_subdir:
        out_dir = os.path.join(save_dir, save_name)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = None

    train_t0 = time.perf_counter()
    try:
        # Calibration: forward-only pass on first subgraph to capture loss magnitudes
        print('Calibration pass (no weight update)...')
        model.reset_loss_trackers()
        calib_gid, calib_nd, calib_lm, calib_total = subgraphs[0]
        with torch.no_grad():
            # Use forward_train on first chunk (sqrtn_forward_backward calls .backward(),
            # incompatible with no_grad context)
            calib_num = min(calib_total,
                            calib_total if cmd_args.blksize < 0 else cmd_args.blksize)
            ll, _ = model.forward_train([calib_gid],
                                        node_feats=calib_nd[:calib_num],
                                        num_nodes=calib_num,
                                        label_mask=calib_lm[:calib_num] if calib_lm is not None else None)
            calib_loss = (-ll / calib_num).item()

        calib_cont = abs(model._ll_cont / calib_num) or 1.0
        calib_bin = abs(model._ll_bin / calib_num) or 1.0
        calib_label = abs(model._ll_label / calib_num) or 1.0
        # KL is in calib_loss when VAE is on; strip it out so it isn't baked into w_cont/w_bin/w_label.
        calib_kl_contrib = pipeline_args.kl_weight * abs(model._kl / calib_num) if pipeline_args.vae_feat else 0.0
        calib_struct = abs(calib_loss - calib_cont - calib_bin - calib_label - calib_kl_contrib) or 1.0

        model.w_cont = (calib_struct / calib_cont) * user_w_cont
        model.w_bin = (calib_struct / calib_bin) * user_w_cont  # binary shares cont weight
        model.w_label = (calib_struct / calib_label) * user_w_label

        print(f'Calibration magnitudes — struct: {calib_struct:.4f}, '
              f'cont: {calib_cont:.4f}, bin: {calib_bin:.4f}, label: {calib_label:.4f}')
        print(f'Dynamic weights — w_cont: {model.w_cont:.4f}, w_bin: {model.w_bin:.4f}, w_label: {model.w_label:.4f}')

        # Seed EMAs from calibration for periodic recalibration (active when recal_momentum < 1.0)
        ema_struct = calib_struct
        ema_cont   = calib_cont
        ema_bin    = calib_bin
        ema_label  = calib_label

        pbar = tqdm(range(cmd_args.num_epochs))
        print(f'Start learn ({len(subgraphs)} subgraph(s), loss weights relative to struct: cont={user_w_cont}, label={user_w_label})')
        for epoch in pbar:
            # KL annealing: update the effective coefficient before any forward pass this epoch.
            # base_kl_weight is the schedule's β=1 ceiling (the user's -kl_weight CLI value).
            kl_beta = compute_kl_beta(epoch, pipeline_args.kl_schedule,
                                      pipeline_args.kl_anneal_epochs,
                                      pipeline_args.kl_cycle_epochs,
                                      pipeline_args.kl_ramp_ratio)
            model.kl_weight = kl_beta * base_kl_weight

            # Anneal scheduled sampling probability linearly from 0 to ss_max_prob
            if epoch < ss_start_epoch or ss_max_prob <= 0:
                model.ss_prob = 0.0
            else:
                model.ss_prob = ss_max_prob * (epoch - ss_start_epoch) / ss_ramp_epochs

            epoch_loss_val = 0.0
            epoch_ll_cont = 0.0
            epoch_ll_bin = 0.0
            epoch_ll_label = 0.0
            epoch_kl = 0.0
            epoch_nodes = 0
            epoch_grad_total = 0.0
            epoch_grad_feat = 0.0
            epoch_grad_label = 0.0
            epoch_grad_state = 0.0
            epoch_grad_struct = 0.0

            for gid, sub_nd, sub_lm, sub_num_nodes in subgraphs:
                optimizer.zero_grad()
                model.reset_loss_trackers()

                if cmd_args.blksize < 0 or sub_num_nodes <= cmd_args.blksize:
                    # Full pass
                    ll, _ = model.forward_train([gid], node_feats=sub_nd, label_mask=sub_lm)
                    loss = -ll / sub_num_nodes
                    loss.backward()
                    sg_loss_val = loss.item()
                else:
                    # Chunked pass
                    ll, _ = sqrtn_forward_backward(model,
                                                   graph_ids=[gid],
                                                   list_node_starts=[0],
                                                   num_nodes=sub_num_nodes,
                                                   blksize=cmd_args.blksize,
                                                   loss_scale=1.0/sub_num_nodes,
                                                   node_feats=sub_nd,
                                                   label_mask=sub_lm)
                    sg_loss_val = -ll / sub_num_nodes

                # Gradient-norm bookkeeping (before clip so we see the true magnitude).
                grad_groups = _group_grad_norms()
                import math as _m
                sg_total_grad = _m.sqrt(sum(v ** 2 for v in grad_groups.values()))
                epoch_grad_total  += sg_total_grad
                epoch_grad_feat   += grad_groups['feat']
                epoch_grad_label  += grad_groups['label']
                epoch_grad_state  += grad_groups['state']
                epoch_grad_struct += grad_groups['struct']

                # Per-group gradient clipping: a struct spike no longer zeroes the
                # feat/label/state heads. Each group is clipped to the same max_norm.
                if cmd_args.grad_clip > 0:
                    for _params in _group_params().values():
                        if _params:
                            torch.nn.utils.clip_grad_norm_(_params, max_norm=cmd_args.grad_clip)
                optimizer.step()

                epoch_loss_val += sg_loss_val
                epoch_ll_cont  += model._ll_cont
                epoch_ll_bin   += model._ll_bin
                epoch_ll_label += model._ll_label
                epoch_kl       += model._kl
                epoch_nodes    += sub_num_nodes

            # Epoch-level monitoring (averaged across subgraphs)
            avg_loss    = epoch_loss_val / len(subgraphs)
            # Per-component trackers store UNWEIGHTED NLLs (customized_models.py applies
            # w_cont/w_bin/w_label only to the aggregated loss before backward).
            ll_cont_val  = -epoch_ll_cont  / epoch_nodes
            ll_bin_val   = -epoch_ll_bin   / epoch_nodes
            ll_label_val = -epoch_ll_label / epoch_nodes
            kl_val       = epoch_kl / epoch_nodes
            # KL adds kl_weight*kl_val to the displayed loss; strip it out for the structure residual.
            # Use model.kl_weight (current effective β·base) so the residual is honest under annealing.
            kl_loss_contrib = model.kl_weight * kl_val if pipeline_args.vae_feat else 0.0
            # Back out current weights to recover unweighted struct (avg_loss is weighted).
            ll_struct_val = (avg_loss
                             - model.w_cont  * ll_cont_val
                             - model.w_bin   * ll_bin_val
                             - model.w_label * ll_label_val
                             - kl_loss_contrib)

            # Periodic EMA recalibration of loss weights (gated on momentum < 1.0).
            # KL stays outside — it's annealed separately via kl_schedule.
            m = pipeline_args.recal_momentum
            if m < 1.0:
                cur_cont   = abs(ll_cont_val)  or 1.0
                cur_bin    = abs(ll_bin_val)   or 1.0
                cur_label  = abs(ll_label_val) or 1.0
                cur_struct = abs(ll_struct_val) or 1.0

                ema_struct = m * ema_struct + (1.0 - m) * cur_struct
                ema_cont   = m * ema_cont   + (1.0 - m) * cur_cont
                ema_bin    = m * ema_bin    + (1.0 - m) * cur_bin
                ema_label  = m * ema_label  + (1.0 - m) * cur_label

                model.w_cont  = (ema_struct / ema_cont)  * user_w_cont
                model.w_bin   = (ema_struct / ema_bin)   * user_w_cont
                model.w_label = (ema_struct / ema_label) * user_w_label

            # Track memory usage
            ram_mb = process.memory_info().rss / 1024 ** 2
            peak_ram_mb = max(peak_ram_mb, ram_mb)
            if gpu_available:
                vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                peak_vram_mb = max(peak_vram_mb, vram_mb)

            bin_desc = f" | bin: {ll_bin_val:.4f}" if model.bin_feat_dim > 0 else ""
            kl_desc = f" | β: {kl_beta:.2f} kl: {kl_val:.4f}" if pipeline_args.vae_feat else ""
            recal_desc = (f" | w: c={model.w_cont:.2f} b={model.w_bin:.2f} l={model.w_label:.2f}"
                          if pipeline_args.recal_momentum < 1.0 else "")
            n_sg = len(subgraphs)
            grad_desc = (f" | ‖g‖: {epoch_grad_total / n_sg:.2e} "
                         f"(feat {epoch_grad_feat / n_sg:.1e}, "
                         f"label {epoch_grad_label / n_sg:.1e}, "
                         f"state {epoch_grad_state / n_sg:.1e}, "
                         f"struct {epoch_grad_struct / n_sg:.1e})")
            pbar.set_description(
                f"Loss: {avg_loss:.4f} | struct: {ll_struct_val:.4f} | cont: {ll_cont_val:.4f}{bin_desc} | label: {ll_label_val:.4f}{kl_desc}{recal_desc}{grad_desc}"
            )

            if (pipeline_args.save_model
                    and pipeline_args.save_model_every > 0
                    and (epoch + 1) % pipeline_args.save_model_every == 0):
                snap_path = (os.path.join(out_dir, f'model_epoch{epoch + 1}.pt')
                             if use_subdir
                             else os.path.join(save_dir, save_name + f'_model_epoch{epoch + 1}.pt'))
                torch.save({'state_dict': model.state_dict(),
                            'cmd_args': vars(cmd_args),
                            'pipeline_args': vars(pipeline_args),
                            'feat_dim': feat_dim,
                            'num_classes': num_classes,
                            'binary_idx': binary_idx,
                            'epoch': epoch + 1},
                           snap_path)

        timing_state['train_seconds'] = time.perf_counter() - train_t0
    except Exception as exc:
        timing_state['train_seconds'] = time.perf_counter() - train_t0
        _capture_peak_mem()
        _record_failure('training', exc)
        if pipeline_args.timing_log_path:
            _write_timing_log()
            print(f'[capacity] training phase failed ({timing_state["status"]}); '
                  f'JSON written to {pipeline_args.timing_log_path}.', file=sys.stderr)
            sys.exit(2)
        raise

    _capture_peak_mem()

    print(f'\n=== Memory usage (training) ===')
    print(f'Peak RAM:  {peak_ram_mb:.1f} MB')
    if gpu_available:
        print(f'Peak VRAM: {peak_vram_mb:.1f} MB')

    model.eval()
    print('Start generate')

    n_gen = (pipeline_args.num_gen_subgraphs
             if pipeline_args.num_gen_subgraphs is not None
             else len(subgraphs))
    timing_state['num_gen_subgraphs'] = n_gen
    gen_t0 = time.perf_counter()
    try:
        if pipeline_args.subsample or pipeline_args.load_subsamples:
            # Generate one synthetic subgraph per training subgraph, save to a directory.
            # out_dir was pre-created before the training loop.
            with torch.no_grad():
                for i, (_, _, _, sub_num_nodes) in enumerate(subgraphs[:n_gen]):
                    _, pred_edges, _, pred_node_feats, _ = model(sub_num_nodes, display=(i == 0))

                    gen_cont_feats = pred_node_feats[:, :feat_dim]
                    gen_labels = pred_node_feats[:, feat_dim:]

                    gen_nx = nx.Graph()
                    gen_nx.add_nodes_from(range(sub_num_nodes))
                    for edge in pred_edges:
                        gen_nx.add_edge(edge[0], edge[1])

                    gen_dgl = build_generated_dgl(gen_nx, graph,
                                                  features=gen_cont_feats,
                                                  labels=gen_labels)
                    dgl.save_graphs(os.path.join(out_dir, f'subgraph_{i}'), [gen_dgl])
                    print(f'  Saved subgraph {i} ({sub_num_nodes} nodes)')

            # Persist the real training subsamples alongside the synthetic outputs
            # so EDA and the future training-source ablation can load them directly.
            train_dir = os.path.join(out_dir, 'training_subsamples')
            os.makedirs(train_dir, exist_ok=True)
            for i, (sg_nx, sub_nd, orig_idx) in enumerate(raw_partitions):
                train_dgl = dgl.from_networkx(sg_nx)
                sub_nd_cpu = sub_nd.detach().cpu()
                train_dgl.ndata['feature'] = sub_nd_cpu[:, :feat_dim]
                train_dgl.ndata['label'] = sub_nd_cpu[:, feat_dim].long()
                train_dgl.ndata['original_indices'] = orig_idx.cpu().long()
                dgl.save_graphs(os.path.join(train_dir, f'subgraph_{i}'), [train_dgl])
            print(f'  Saved {len(raw_partitions)} real training subsamples to {train_dir}')

            if norm_stats is not None:
                torch.save(norm_stats, os.path.join(out_dir, 'norm_stats.pt'))
            if binary_idx:
                torch.save(binary_idx, os.path.join(out_dir, 'binary_idx.pt'))
            if pipeline_args.cat_feat:
                torch.save({'bin_edges': bin_edges,
                            'bin_centers': bin_centers,
                            'bin_sigma': pipeline_args.bin_sigma,
                            'n_bins': pipeline_args.n_bins},
                           os.path.join(out_dir, 'cat_bins.pt'))
            if pipeline_args.save_model:
                torch.save({'state_dict': model.state_dict(),
                            'cmd_args': vars(cmd_args),
                            'pipeline_args': vars(pipeline_args),
                            'feat_dim': feat_dim,
                            'num_classes': num_classes,
                            'binary_idx': binary_idx},
                           os.path.join(out_dir, 'model.pt'))

        else:
            # Single full graph — original code path
            with torch.no_grad():
                target_num_nodes = graph_nx.number_of_nodes()
                _, pred_edges, _, pred_node_feats, _ = model(target_num_nodes, display=True)

            gen_cont_feats = pred_node_feats[:, :feat_dim]
            gen_labels = pred_node_feats[:, feat_dim:]

            gen_nx = nx.Graph()
            gen_nx.add_nodes_from(range(target_num_nodes))
            for edge in pred_edges:
                gen_nx.add_edge(edge[0], edge[1])

            gen_dgl = build_generated_dgl(gen_nx, graph,
                                          features=gen_cont_feats,
                                          labels=gen_labels)

            # save_dir was pre-created before the training loop.
            dgl.save_graphs(os.path.join(save_dir, save_name), [gen_dgl])

            if norm_stats is not None:
                torch.save(norm_stats, os.path.join(save_dir, save_name + '_norm_stats.pt'))
            if binary_idx:
                torch.save(binary_idx, os.path.join(save_dir, save_name + '_binary_idx.pt'))
            if pipeline_args.cat_feat:
                torch.save({'bin_edges': bin_edges,
                            'bin_centers': bin_centers,
                            'bin_sigma': pipeline_args.bin_sigma,
                            'n_bins': pipeline_args.n_bins},
                           os.path.join(save_dir, save_name + '_cat_bins.pt'))
            if pipeline_args.save_model:
                torch.save({'state_dict': model.state_dict(),
                            'cmd_args': vars(cmd_args),
                            'pipeline_args': vars(pipeline_args),
                            'feat_dim': feat_dim,
                            'num_classes': num_classes,
                            'binary_idx': binary_idx},
                           os.path.join(save_dir, save_name + '_model.pt'))

        timing_state['gen_seconds'] = time.perf_counter() - gen_t0
    except Exception as exc:
        timing_state['gen_seconds'] = time.perf_counter() - gen_t0
        _capture_peak_mem()
        _record_failure('generation', exc)
        if pipeline_args.timing_log_path:
            _write_timing_log()
            print(f'[capacity] generation phase failed ({timing_state["status"]}); '
                  f'JSON written to {pipeline_args.timing_log_path}.', file=sys.stderr)
            sys.exit(2)
        raise

    _capture_peak_mem()
    if pipeline_args.timing_log_path:
        _write_timing_log()


if __name__ == '__main__':
    main()
