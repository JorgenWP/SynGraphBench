import argparse
import time
import logging
from link_utils import *
import pandas
import os
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--val_ratio', type=float, default=0.05)
parser.add_argument('--test_ratio', type=float, default=0.1)
parser.add_argument('--neg_sampling', type=str, default='random',
                    choices=['random', 'hard'],
                    help='Negative sampling strategy: random or hard (2-hop)')
parser.add_argument('--decoder', type=str, default='dot',
                    choices=['dot', 'mlp'],
                    help='Edge decoder: dot product or MLP on Hadamard product')
args = parser.parse_args()

columns = ['name']
datasets = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance',
            'tolokers', 'questions']
models = model_lp_dict.keys()

if args.datasets is not None:
    datasets = args.datasets.split(',')
    print('Evaluated Datasets: ', datasets)

if args.models is not None:
    models = args.models.split(',')
    print('Evaluated Models: ', models)

for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        time_cost = 0
        train_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epochs': 200,
            'patience': 50,
            'metric': 'AUPRC',
            'neg_sampling': args.neg_sampling,
            'decoder': args.decoder,
        }
        data = LinkDataset("original/"+dataset_name)
        model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}

        auc_list, pre_list, rec_list = [], [], []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            data.split(args.val_ratio, args.test_ratio, t, args.neg_sampling)
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            detector = model_lp_dict[model](train_config, model_config, data)
            st = time.time()
            print(detector.model)
            test_score = detector.train()
            auc_list.append(test_score['AUROC'])
            pre_list.append(test_score['AUPRC'])
            rec_list.append(test_score['RecK'])
            ed = time.time()
            time_cost += ed - st
        del detector, data

        model_result[dataset_name+'-AUROC mean'] = np.mean(auc_list)
        model_result[dataset_name+'-AUROC std'] = np.std(auc_list)
        model_result[dataset_name+'-AUPRC mean'] = np.mean(pre_list)
        model_result[dataset_name+'-AUPRC std'] = np.std(pre_list)
        model_result[dataset_name+'-RecK mean'] = np.mean(rec_list)
        model_result[dataset_name+'-RecK std'] = np.std(rec_list)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id)
    print(results)
