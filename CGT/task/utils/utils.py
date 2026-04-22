import csv
import os.path as osp
import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
import pandas as pd
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize
from os.path import exists

import torch
import torch.nn.functional as F

from ogb.io.read_graph_pyg import read_graph_pyg
from torch_geometric.utils import to_undirected


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_ratio = 0.4
val_ratio = 0.2
def split_ids(args, node_num):
    node_ids = list(range(node_num))
    random.shuffle(node_ids)

    ids = {}
    ids['train'] = node_ids[:int(train_ratio * len(node_ids))]
    ids['val'] = node_ids[int(train_ratio * len(node_ids)):int((train_ratio + val_ratio) * len(node_ids))]
    ids['test'] = node_ids[int((train_ratio + val_ratio) * len(node_ids)):]

    return ids


def convert_to_edge_list(edge_index, X):
    edge_list = []
    sorted, indices = torch.sort(edge_index[1])
    source_ids = edge_index[0][indices]
    target_ids = edge_index[1][indices]

    j = 0
    for i in range(X.shape[0]):
        neighbor_list = []
        while j < target_ids.shape[0] and target_ids[j] == i:
            neighbor_list.append(source_ids[j].item())
            j += 1
        edge_list.append(neighbor_list)

    return edge_list


def normalize_features(features):
    features = features - features.min()
    features.div_(features.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return features


def load_ogbn(args):
    master_file = args.data_dir + "/ogbn-master.csv"
    master = pd.read_csv(master_file, index_col = 0)
    meta_dict = master[args.dataset]

    add_inverse_edge = meta_dict['add_inverse_edge'] == 'True'
    binary = meta_dict['binary'] == 'True'
    additional_node_files = []
    additional_edge_files = []

    data_dir = args.data_dir + "/" + args.dataset + "/"
    data = read_graph_pyg(data_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=binary)[0]
    #data.x = normalize_features(data.x)
    node_feat = data.x.numpy()

    data.edge_index = to_undirected(data.edge_index)
    graph = convert_to_edge_list(data.edge_index, node_feat)

    label = pd.read_csv(osp.join(data_dir, 'node-label.csv.gz'), compression='gzip', header = None).values
    label = label.squeeze()

    feat_size = node_feat.shape[1]
    label_size = label.max() - label.min() + 1

    return graph, node_feat, label, feat_size, label_size


def load_dgl_graph(args):
    """Load a DGL binary graph file (GADBench format) and convert to CGT format."""
    from dgl.data.utils import load_graphs

    graph_path = osp.join(args.data_dir, args.dataset)
    graph = load_graphs(graph_path)[0][0]

    # Extract features and labels
    features = graph.ndata['feature'].numpy().astype(np.float32)
    labels = graph.ndata['label'].numpy().astype(np.int64)

    # Normalize features
    features = normalize(features, axis=1, norm='l2')

    # Build adjacency list from DGL edges
    src, dst = graph.edges()
    src, dst = src.numpy(), dst.numpy()
    num_nodes = graph.num_nodes()
    adj_list = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj_list[s].append(int(d))
    # Ensure undirected
    for s, d in zip(dst, src):
        if int(d) not in adj_list[s]:
            adj_list[s].append(int(d))

    feat_size = features.shape[1]
    labels = labels - labels.min()
    label_size = int(labels.max() - labels.min() + 1)

    return adj_list, features, labels, feat_size, label_size


def load_dgl_graph_with_hidden_links(args, trial_id,
                                     val_ratio=0.05, test_ratio=0.10):
    """Load a DGL graph with the trial's test edges stripped from adjacency.

    The edge split mirrors GADBench/link_utils.py:LinkDataset.split(trial_id)
    byte-for-byte so the same edges end up withheld on both sides:
      seed = 3407 + trial_id * 10
      MST edges protected, non-tree candidates shuffled via torch.randperm,
      first int(E * test_ratio) candidates become the test split.

    Val edges are NOT stripped (matches the mask-test-only decision,
    analogous to BiGG's --mask_test_labels).

    Returns:
        adj_list, features, labels, feat_size, label_size, test_edges
        where test_edges is a [n_test, 2] LongTensor of (src, dst) pairs.
    """
    import dgl
    from dgl.data.utils import load_graphs

    graph_path = osp.join(args.data_dir, args.dataset)
    graph = load_graphs(graph_path)[0][0]

    features = graph.ndata['feature'].numpy().astype(np.float32)
    labels = graph.ndata['label'].numpy().astype(np.int64)
    features = normalize(features, axis=1, norm='l2')

    # === Mirror of LinkDataset.split(trial_id). Keep in sync with
    #     GADBench/link_utils.py:LinkDataset.split. ===
    torch.manual_seed(3407 + trial_id * 10)
    src, dst = graph.edges()
    E = src.shape[0]

    nx_graph = dgl.to_networkx(graph).to_undirected()
    tree_edges_nx = set(nx.minimum_spanning_tree(nx_graph).edges())

    tree_mask = torch.zeros(E, dtype=torch.bool)
    for i in range(E):
        u, v = src[i].item(), dst[i].item()
        if (u, v) in tree_edges_nx or (v, u) in tree_edges_nx:
            tree_mask[i] = True

    candidate_idx = torch.where(~tree_mask)[0]
    n_candidates = candidate_idx.shape[0]
    perm = torch.randperm(n_candidates)
    candidate_idx = candidate_idx[perm]

    n_test_target = int(E * test_ratio)
    n_test = min(n_test_target, n_candidates)
    test_idx = candidate_idx[:n_test]

    test_edges = torch.stack([src[test_idx], dst[test_idx]], dim=1)
    # === end mirror ===

    # Strip test edges from adjacency (undirected: both directions)
    keep = torch.ones(E, dtype=torch.bool)
    keep[test_idx] = False
    src_keep = src[keep].numpy()
    dst_keep = dst[keep].numpy()

    num_nodes = graph.num_nodes()
    adj_list = [[] for _ in range(num_nodes)]
    for s, d in zip(src_keep, dst_keep):
        adj_list[int(s)].append(int(d))
    for s, d in zip(dst_keep, src_keep):
        if int(d) not in adj_list[int(s)]:
            adj_list[int(s)].append(int(d))

    feat_size = features.shape[1]
    labels = labels - labels.min()
    label_size = int(labels.max() - labels.min() + 1)

    print(f"hidden_links split trial={trial_id}: "
          f"{E} edges, stripped {n_test} test edges "
          f"(target {n_test_target}), kept {int(keep.sum())}")

    return adj_list, features, labels, feat_size, label_size, test_edges


def split_node_ids_for_hidden_links(num_nodes, trial_id, val_fraction=0.2):
    """Deterministic 80/20 node split for CGT GPT training under hidden_links.

    Link prediction has no inherent node-level split (all nodes are
    observed), but CGT's generator needs two disjoint target_ids buckets
    so both gen_train_ids and gen_val_ids are non-empty and together
    cover every node. Offset the seed from the edge-split seed so the
    two splits are uncorrelated.
    """
    rng = np.random.RandomState(3407 + trial_id * 10 + 1)
    perm = rng.permutation(num_nodes)
    n_val = max(1, int(num_nodes * val_fraction))
    return {
        'train': perm[n_val:].tolist(),
        'val': perm[:n_val].tolist(),
        'test': [],
    }


def split_ids_from_dgl(args, semi_supervised=False, trial_id=0):
    """Extract pre-defined train/val/test splits from a DGL graph (GADBench format)."""
    from dgl.data.utils import load_graphs

    graph_path = osp.join(args.data_dir, args.dataset)
    graph = load_graphs(graph_path)[0][0]

    if semi_supervised:
        trial_id += 10

    train_mask = graph.ndata['train_masks'][:, trial_id].numpy().astype(bool)
    val_mask = graph.ndata['val_masks'][:, trial_id].numpy().astype(bool)
    test_mask = graph.ndata['test_masks'][:, trial_id].numpy().astype(bool)

    ids = {
        'train': np.where(train_mask)[0].tolist(),
        'val': np.where(val_mask)[0].tolist(),
        'test': np.where(test_mask)[0].tolist(),
    }
    return ids


def load_graph(args):
    if args.dataset in ("ogbn-arxiv", "ogbn-products"):
        return load_ogbn(args)

    # Check for DGL binary graph file (GADBench format)
    dgl_path = osp.join(args.data_dir, args.dataset)
    if osp.isfile(dgl_path) and not dgl_path.endswith('.npz'):
        return load_dgl_graph(args)

    dataset = args.data_dir + "/" + args.dataset + ".npz"
    with np.load(dataset, allow_pickle = True) as loader:
        loader = dict(loader)

        # Adjacency matrix
        graph = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']), shape=loader['adj_shape'])
        graph = graph + graph.transpose()
        if args.noise_num > 0:
            graph = graph + sp.identity(graph.shape[0])
        graph = sp.csr_matrix.toarray(graph)

        # Feature matrix
        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']), shape=loader['attr_shape'])
            features = sp.csr_matrix.toarray(features)
            # Normalize
            features = normalize(features, axis=1, norm='l2')
            #features = features - np.mean(features, axis=0)
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            features = loader['attr_matrix']
        else:
            features = None

        # Labels
        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']), shape=loader['labels_shape'])
            labels = sp.csr_matrix.toarray(labels)
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

    feat_size = features.shape[1]
    labels = labels - labels.min()
    label_size = labels.max() - labels.min() + 1

    return graph, features, labels, feat_size, label_size


def calc_loss(y_pred, y_true):
    if len(y_pred.shape) == 1:
        y_pred = torch.unsqueeze(y_pred, 0)
    if len(y_true.shape) == 2:
        y_true = torch.squeeze(y_true)
    loss_train = F.cross_entropy(y_pred, y_true)
    return loss_train


def calc_f1(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    y_true = y_true.cpu()
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")


