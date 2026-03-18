import os
import dgl
import torch
import torch.optim as optim
import networkx as nx
from tqdm import tqdm

from bigg.common.configs import cmd_args, set_device
from bigg.model.tree_clib.tree_lib import setup_treelib, TreeLib
from bigg.model.tree_model import RecurTreeGen
from bigg.experiments.train_utils import sqrtn_forward_backward


def main():
    set_device(cmd_args.gpu)
    setup_treelib(cmd_args)

    # Load dataset
    DATASET = cmd_args.data_dir
    graphs_dgl, _ = dgl.load_graphs('../datasets/original/' + DATASET)
    graph = graphs_dgl[0]

    # Convert topology for TreeLib (structure only — no features)
    graph_nx = nx.Graph(graph.to_networkx().to_undirected())
    graph_nx.remove_edges_from(nx.selfloop_edges(graph_nx))

    TreeLib.InsertGraph(graph_nx)

    num_nodes = graph_nx.number_of_nodes()
    num_edges = graph_nx.number_of_edges()
    print(f'Dataset: {DATASET}  |  nodes: {num_nodes}  |  edges: {num_edges}')

    cmd_args.has_node_feats = False
    cmd_args.has_edge_feats = False
    cmd_args.max_num_nodes = num_nodes

    model = RecurTreeGen(cmd_args).to(cmd_args.device)
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate, weight_decay=1e-4)

    indices = [0]

    model.train()
    print('Start training (structure only)')
    pbar = tqdm(range(cmd_args.num_epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        if cmd_args.blksize < 0 or num_nodes <= cmd_args.blksize:
            ll, _ = model.forward_train(indices)
            loss = -ll / num_nodes
            loss.backward()
            loss_val = loss.item()
        else:
            ll, _ = sqrtn_forward_backward(model,
                                           graph_ids=indices,
                                           list_node_starts=[0],
                                           num_nodes=num_nodes,
                                           blksize=cmd_args.blksize,
                                           loss_scale=1.0 / num_nodes)
            loss_val = -ll / num_nodes

        if cmd_args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)

        optimizer.step()
        pbar.set_description(f'Loss: {loss_val:.4f}')

    # Save checkpoint
    os.makedirs(cmd_args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cmd_args.save_dir, f'bigg_structure_{DATASET}.ckpt')
    torch.save(model.state_dict(), ckpt_path)
    print(f'Checkpoint saved to {ckpt_path}')

    model.eval()
    print('Start generation')
    with torch.no_grad():
        _, pred_edges, _, _, _ = model(num_nodes)

    gen_nx = nx.Graph()
    gen_nx.add_nodes_from(range(num_nodes))
    gen_nx.add_edges_from(pred_edges)

    gen_num_edges = gen_nx.number_of_edges()
    print(f'Generated graph  |  nodes: {num_nodes}  |  edges: {gen_num_edges}  '
          f'(original: {num_edges})')

    # Build DGL graph and attach placeholder features/labels so it is
    # compatible with the rest of the benchmark pipeline
    gen_dgl = dgl.from_networkx(gen_nx)

    feat_dim = graph.ndata['feature'].shape[1]
    num_splits = graph.ndata['train_masks'].shape[1]

    gen_dgl.ndata['feature'] = torch.zeros(num_nodes, feat_dim)
    gen_dgl.ndata['label'] = torch.zeros(num_nodes, dtype=torch.long)
    gen_dgl.ndata['train_masks'] = torch.ones(num_nodes, num_splits, dtype=torch.uint8)
    gen_dgl.ndata['val_masks'] = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)
    gen_dgl.ndata['test_masks'] = torch.zeros(num_nodes, num_splits, dtype=torch.uint8)

    save_name = f'{DATASET}_structure_blksize_{cmd_args.blksize}_lr_{cmd_args.learning_rate}_epochs_{cmd_args.num_epochs}'
    save_dir = '../datasets/synthetic/bigg'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    dgl.save_graphs(save_path, [gen_dgl])
    print(f'Saved generated graph to {save_path}')


if __name__ == '__main__':
    main()
