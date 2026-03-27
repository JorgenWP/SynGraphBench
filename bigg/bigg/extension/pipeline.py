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
from bigg.extension.customized_models import BiggWithConditionedFeats, BiggWithFeatsAndLabels

# Preprocessing utilities
from bigg.extension.preprocessing import (
    load_dgl_graph,
    dgl_to_networkx,
    bfs_reorder,
    normalize_features,
    build_generated_dgl,
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

    pipeline_args, _ = pipeline_parser.parse_known_args()

    # Parse loss weights
    lw = [float(x) for x in pipeline_args.loss_weights.split(',')]
    assert len(lw) == 2, f'Expected 2 loss weights (cont,label), got {len(lw)}'
    user_w_cont, user_w_label = lw

    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)

    #Load dataset
    DATASET = cmd_args.data_dir
    graph = load_dgl_graph(DATASET)

    #Get features and labels
    cont_feats = graph.ndata['feature']
    labels = graph.ndata['label']

    # Optionally normalise features
    if pipeline_args.normalize is not None:
        cont_feats = normalize_features(cont_feats, pipeline_args.normalize)
        print(f'Applied {pipeline_args.normalize} normalisation to features')

    #Determine dimensions of features
    feat_dim = cont_feats.shape[1]
    num_classes = int(labels.max().item()) + 1

    node_data = torch.cat([cont_feats, labels.unsqueeze(1).float()], dim = 1).to(cmd_args.device)
    list_node_feats = [node_data]

    #Convert topology for treelib
    graph_nx = dgl_to_networkx(graph)

    # Apply fixed BFS node ordering: reorder graph and features so that
    # BFS-adjacent nodes have consecutive indices
    if pipeline_args.bfs_preprocess:
        graph_nx, node_data = bfs_reorder(graph_nx, node_data)
        list_node_feats = [node_data]

    TreeLib.InsertGraph(graph_nx)

    cmd_args.has_node_feats = True
    cmd_args.max_num_nodes = graph_nx.number_of_nodes()

    if pipeline_args.model_type == 'conditional':
        model = BiggWithConditionedFeats(cmd_args, feat_dim=feat_dim, num_classes=num_classes,
                                         label_temp=pipeline_args.label_temp,
                                         noise_std=pipeline_args.noise_std,
                                         hetero_feat=pipeline_args.hetero_feat).to(cmd_args.device)
    elif pipeline_args.model_type == 'independent':
        model = BiggWithFeatsAndLabels(cmd_args, feat_dim=feat_dim, num_classes=num_classes,
                                       label_temp=pipeline_args.label_temp,
                                       noise_std=pipeline_args.noise_std,
                                       hetero_feat=pipeline_args.hetero_feat).to(cmd_args.device)

    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)

    model.train()

    indices = [0]
    num_nodes = graph_nx.number_of_nodes()

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

    # Calibration: forward-only pass to capture loss magnitudes for dynamic normalization
    print('Calibration pass (no weight update)...')
    model.reset_loss_trackers()
    batch_node_feats = torch.cat([list_node_feats[i] for i in indices], dim=0)

    with torch.no_grad():
        if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
            ll, _ = model.forward_train(indices, node_feats=batch_node_feats)
            calib_loss = (-ll / num_nodes).item()
        else:
            ll, _ = sqrtn_forward_backward(model,
                                           graph_ids=indices,
                                           list_node_starts=[0],
                                           num_nodes=num_nodes,
                                           blksize=cmd_args.blksize,
                                           loss_scale=1.0/num_nodes,
                                           node_feats=batch_node_feats)
            calib_loss = -ll / num_nodes

    calib_cont = abs(model._ll_cont / num_nodes) or 1.0
    calib_label = abs(model._ll_label / num_nodes) or 1.0
    calib_struct = abs(calib_loss - calib_cont - calib_label) or 1.0

    model.w_cont = (calib_struct / calib_cont) * user_w_cont
    model.w_label = (calib_struct / calib_label) * user_w_label

    print(f'Calibration magnitudes — struct: {calib_struct:.4f}, '
          f'cont: {calib_cont:.4f}, label: {calib_label:.4f}')
    print(f'Dynamic weights — w_cont: {model.w_cont:.4f}, w_label: {model.w_label:.4f}')

    pbar = tqdm(range(cmd_args.num_epochs))
    print(f'Start learn (loss weights relative to struct: cont={user_w_cont}, label={user_w_label})')
    for epoch in pbar:
        # Anneal scheduled sampling probability linearly from 0 to ss_max_prob
        if epoch < ss_start_epoch or ss_max_prob <= 0:
            model.ss_prob = 0.0
        else:
            model.ss_prob = ss_max_prob * (epoch - ss_start_epoch) / ss_ramp_epochs

        optimizer.zero_grad()
        model.reset_loss_trackers()
        batch_node_feats = torch.cat([list_node_feats[i] for i in indices], dim=0)

        # Gradient Checkpointing Logic
        if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
            # Full pass (for small graphs)
            ll, _ = model.forward_train(indices, node_feats=batch_node_feats)
            loss = -ll / num_nodes
            loss.backward()
            loss_val = loss.item()
        else:
            # Chunked pass (for large graphs like Tolokers)
            # This calls .backward() internally for each chunk to save VRAM
            ll, _ = sqrtn_forward_backward(model,
                                           graph_ids=indices,
                                           list_node_starts=[0],
                                           num_nodes=num_nodes,
                                           blksize=cmd_args.blksize,
                                           loss_scale=1.0/num_nodes,
                                           node_feats=batch_node_feats)
            loss_val = -ll / num_nodes

        if cmd_args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)

        optimizer.step()

        # Compute per-node component losses (unweighted, for monitoring)
        ll_cont_val = -model._ll_cont / num_nodes
        ll_label_val = -model._ll_label / num_nodes
        ll_struct_val = loss_val - ll_cont_val - ll_label_val

        # Track memory usage
        ram_mb = process.memory_info().rss / 1024 ** 2
        peak_ram_mb = max(peak_ram_mb, ram_mb)
        if gpu_available:
            vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            peak_vram_mb = max(peak_vram_mb, vram_mb)

        pbar.set_description(
            f"Loss: {loss_val:.4f} | struct: {ll_struct_val:.4f} | cont: {ll_cont_val:.4f} | label: {ll_label_val:.4f}"
        )

    print(f'\n=== Memory usage (training) ===')
    print(f'Peak RAM:  {peak_ram_mb:.1f} MB')
    if gpu_available:
        print(f'Peak VRAM: {peak_vram_mb:.1f} MB')

    model.eval()
    print('Start generate')
    with torch.no_grad():
        target_num_nodes = graph_nx.number_of_nodes()

        #Generating graph
        _, pred_edges, _, pred_node_feats, _ = model(
            target_num_nodes, display=True
        )

    gen_cont_feats = pred_node_feats[:, :feat_dim]
    gen_labels = pred_node_feats[:, feat_dim:]

    # Build DGL graph with masks
    gen_nx = nx.Graph()
    gen_nx.add_nodes_from(range(target_num_nodes))
    for edge in pred_edges:
        gen_nx.add_edge(edge[0], edge[1])

    gen_dgl = build_generated_dgl(gen_nx, graph,
                                  features=gen_cont_feats,
                                  labels=gen_labels)

    # Save generated graph
    norm_tag = pipeline_args.normalize if pipeline_args.normalize is not None else 'none'
    bfs_tag = 'bfs' if pipeline_args.bfs_preprocess else 'nobfs'
    lw_tag = pipeline_args.loss_weights.replace(',', '_')
    hetero_tag = 'hetero' if pipeline_args.hetero_feat else 'det'
    save_name = f'blksize_{cmd_args.blksize}_b_{cmd_args.batch_size}_lr_{cmd_args.learning_rate}_epochs_{cmd_args.num_epochs}_noise_{pipeline_args.noise_std}_ss_{pipeline_args.ss_max_prob}_norm_{norm_tag}_{bfs_tag}_lw_{lw_tag}_{hetero_tag}'
    save_dir = f'../datasets/synthetic/bigg/{DATASET}/hidden_labels'
    os.makedirs(save_dir, exist_ok=True)
    dgl.save_graphs(os.path.join(save_dir, save_name), [gen_dgl])


if __name__ == '__main__':
    main()
