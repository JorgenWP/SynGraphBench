import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from datetime import datetime
from time import perf_counter

from args import get_args, get_parser, print_args, print_non_default_args
from task.utils.utils import load_graph, split_ids, split_ids_from_dgl

import generator.gpt.gpt as gpt


def is_dgl_dataset(args):
    """Check if the dataset is a DGL binary graph file (GADBench format)."""
    import os.path as osp
    dgl_path = osp.join(args.data_dir, args.dataset)
    return osp.isfile(dgl_path) and not dgl_path.endswith('.npz')


def main():
    args = get_args()
    args.gpt_train_name = (
        f"{args.dataset}_{args.gpt_model}"
        f"_e{args.gpt_epochs}_l{args.gpt_layers}_h{args.gpt_heads}"
        f"_c{args.cluster_num}_s{args.cg_depth}x{args.cg_fanout}"
    )

    print(f"--- Training CGT on {args.dataset} dataset ---")

    print_args(args)

    print_non_default_args(args)

    # Load graph dataset
    adj, feat, label, feat_size, label_size = load_graph(args)
    args.feat_size = feat_size
    args.label_size = label_size
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use pre-defined splits for DGL/GADBench datasets, random splits otherwise
    if is_dgl_dataset(args):
        ids = split_ids_from_dgl(args)
        print(f"Using GADBench splits: train={len(ids['train'])}, val={len(ids['val'])}, test={len(ids['test'])}")
    else:
        ids = split_ids(args, feat.shape[0])

    # Train CGT and generate synthetic train/val data
    print("\nTraining CGT and generating synthetic data...")
    start_time = perf_counter()
    result = gpt.train_and_generate(args, adj, feat, label, ids)
    print('Total CGT time: {:.3f}'.format(perf_counter() - start_time))

    # Save synthetic data
    save_dir = os.path.join(args.data_dir, '..', 'synthetic')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"cgt_{args.dataset}.pt")

    torch.save({
        **result,
        'ids': ids,
        'feat_size': feat_size,
        'label_size': label_size,
        'cg_depth': args.cg_depth,
        'cg_fanout': args.cg_fanout,
        'noise_num': args.noise_num,
        'self_connection': args.self_connection,
    }, save_path)

    print(f"\nSynthetic dataset saved to {save_path}")


if __name__ == "__main__":
    main()
