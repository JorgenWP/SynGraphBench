import os
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
    pipeline_parser.add_argument('--mask_test_labels', action='store_true', default=False,
                                 help='Exclude test node labels (split 0) from label loss to prevent '
                                      'data leakage in anomaly detection benchmarks')
    pipeline_parser.add_argument('--subsample', action='store_true', default=False,
                                 help='Sample subgraphs via forest fire for VRAM-limited training')
    pipeline_parser.add_argument('-subsample_size', type=int, default=2000,
                                 help='Target number of nodes per subgraph (default: 2000)')
    pipeline_parser.add_argument('-burn_prob', type=float, default=0.3,
                                 help='Forest fire burn probability — controls subgraph density (default: 0.3)')
    pipeline_parser.add_argument('-num_subgraphs', type=int, default=None,
                                 help='Number of subgraphs to generate. Default: ceil(N / subsample_size)')

    pipeline_args, _ = pipeline_parser.parse_known_args()

    # Parse loss weights
    lw = [float(x) for x in pipeline_args.loss_weights.split(',')]
    assert len(lw) == 2, f'Expected 2 loss weights (cont,label), got {len(lw)}'
    user_w_cont, user_w_label = lw

    # AR categorical is its own predictor; reject combinations that don't apply.
    if pipeline_args.cat_feat and (pipeline_args.hetero_feat or pipeline_args.vae_feat):
        raise ValueError('--cat_feat is mutually exclusive with --hetero_feat and --vae_feat')

    # MDN replaces the continuous head entirely; reject combinations that don't apply.
    if pipeline_args.mdn_feat and (pipeline_args.hetero_feat or pipeline_args.vae_feat
                                    or pipeline_args.cat_feat):
        raise ValueError('--mdn_feat is mutually exclusive with --hetero_feat, --vae_feat, --cat_feat')

    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)

    #Load dataset
    DATASET = cmd_args.data_dir
    graph = load_dgl_graph(DATASET)

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

    # --- Build list of (gid, sub_node_data, sub_label_mask, sub_num_nodes) ---
    # When subsampling is disabled this list has a single entry for the full graph.
    if pipeline_args.subsample:
        import math
        n_total = graph_nx.number_of_nodes()
        num_subgraphs = pipeline_args.num_subgraphs or math.ceil(n_total / pipeline_args.subsample_size)
        print(f'Subsampling enabled: {num_subgraphs} forest fire samples, '
              f'target_size={pipeline_args.subsample_size}, burn_prob={pipeline_args.burn_prob}')
        raw_partitions = []
        for _ in range(num_subgraphs):
            raw_partitions.append(forest_fire_subsample(
                graph_nx, node_data,
                target_size=pipeline_args.subsample_size,
                burn_prob=pipeline_args.burn_prob,
            ))
        print(f'  Produced {len(raw_partitions)} samples '
              f'(sizes: {[sg.number_of_nodes() for sg, _, _ in raw_partitions]})')

        subgraphs = []  # list of (gid, sub_node_data, sub_label_mask, sub_num_nodes)
        for sg, sub_nd, orig_idx in raw_partitions:
            if pipeline_args.bfs_preprocess:
                sg, sub_nd, perm = bfs_reorder(sg, sub_nd)
                sub_lm = label_mask[orig_idx[perm]] if label_mask is not None else None
            else:
                sub_lm = label_mask[orig_idx] if label_mask is not None else None
            gid = TreeLib.InsertGraph(sg)
            subgraphs.append((gid, sub_nd.to(cmd_args.device), sub_lm, sg.number_of_nodes()))
    else:
        # Single full graph
        if pipeline_args.bfs_preprocess:
            graph_nx, node_data, perm = bfs_reorder(graph_nx, node_data)
            list_node_feats = [node_data]
            if label_mask is not None:
                label_mask = label_mask[perm]
        TreeLib.InsertGraph(graph_nx)
        subgraphs = [(0, node_data, label_mask, graph_nx.number_of_nodes())]

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
                                 binary_idx=binary_idx).to(cmd_args.device)
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
                                         kl_weight=pipeline_args.kl_weight).to(cmd_args.device)
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
                                       kl_weight=pipeline_args.kl_weight).to(cmd_args.device)

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)

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

    pbar = tqdm(range(cmd_args.num_epochs))
    print(f'Start learn ({len(subgraphs)} subgraph(s), loss weights relative to struct: cont={user_w_cont}, label={user_w_label})')
    for epoch in pbar:
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

            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)
            optimizer.step()

            epoch_loss_val += sg_loss_val
            epoch_ll_cont  += model._ll_cont
            epoch_ll_bin   += model._ll_bin
            epoch_ll_label += model._ll_label
            epoch_kl       += model._kl
            epoch_nodes    += sub_num_nodes

        # Epoch-level monitoring (averaged across subgraphs)
        avg_loss    = epoch_loss_val / len(subgraphs)
        ll_cont_val  = -epoch_ll_cont  / epoch_nodes
        ll_bin_val   = -epoch_ll_bin   / epoch_nodes
        ll_label_val = -epoch_ll_label / epoch_nodes
        kl_val       = epoch_kl / epoch_nodes
        # KL adds kl_weight*kl_val to the displayed loss; strip it out for the structure residual.
        kl_loss_contrib = pipeline_args.kl_weight * kl_val if pipeline_args.vae_feat else 0.0
        ll_struct_val = avg_loss - ll_cont_val - ll_bin_val - ll_label_val - kl_loss_contrib

        # Track memory usage
        ram_mb = process.memory_info().rss / 1024 ** 2
        peak_ram_mb = max(peak_ram_mb, ram_mb)
        if gpu_available:
            vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            peak_vram_mb = max(peak_vram_mb, vram_mb)

        bin_desc = f" | bin: {ll_bin_val:.4f}" if model.bin_feat_dim > 0 else ""
        kl_desc = f" | kl: {kl_val:.4f}" if pipeline_args.vae_feat else ""
        pbar.set_description(
            f"Loss: {avg_loss:.4f} | struct: {ll_struct_val:.4f} | cont: {ll_cont_val:.4f}{bin_desc} | label: {ll_label_val:.4f}{kl_desc}"
        )

    print(f'\n=== Memory usage (training) ===')
    print(f'Peak RAM:  {peak_ram_mb:.1f} MB')
    if gpu_available:
        print(f'Peak VRAM: {peak_vram_mb:.1f} MB')

    # Build save name and directory
    norm_tag = pipeline_args.normalize if pipeline_args.normalize is not None else 'none'
    bfs_tag = 'bfs' if pipeline_args.bfs_preprocess else 'nobfs'
    lw_tag = pipeline_args.loss_weights.replace(',', '_')
    hetero_tag = 'hetero' if pipeline_args.hetero_feat else 'det'
    lvf_tag = f'_lvf{pipeline_args.logvar_floor}' if pipeline_args.hetero_feat else ''
    mask_tag = '_masked' if pipeline_args.mask_test_labels else ''
    bin_tag = '_binfeat' if pipeline_args.binary_feat else ''
    vae_tag = f'_vae{pipeline_args.vae_dim}_kl{pipeline_args.kl_weight}' if pipeline_args.vae_feat else ''
    cat_tag = f'_cat{pipeline_args.n_bins}_smed{pipeline_args.bin_sigma.median().item():.2f}' if pipeline_args.cat_feat else ''
    mdn_tag = f'_mdn{pipeline_args.mdn_components}' if pipeline_args.mdn_feat else ''
    sub_tag = f'_sub{len(subgraphs)}_size{pipeline_args.subsample_size}_p{pipeline_args.burn_prob}' if pipeline_args.subsample else ''
    save_name = f'blksize_{cmd_args.blksize}_b_{cmd_args.batch_size}_lr_{cmd_args.learning_rate}_epochs_{cmd_args.num_epochs}_noise_{pipeline_args.noise_std}_ss_{pipeline_args.ss_max_prob}_norm_{norm_tag}_{bfs_tag}_lw_{lw_tag}_{hetero_tag}{lvf_tag}{mask_tag}{bin_tag}{vae_tag}{cat_tag}{mdn_tag}{sub_tag}'
    save_dir = f'../datasets/synthetic/bigg/{DATASET}/hidden_labels'

    model.eval()
    print('Start generate')

    if pipeline_args.subsample:
        # Generate one synthetic subgraph per training subgraph, save to a directory
        out_dir = os.path.join(save_dir, save_name)
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            for i, (_, _, _, sub_num_nodes) in enumerate(subgraphs):
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

        os.makedirs(save_dir, exist_ok=True)
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


if __name__ == '__main__':
    main()
