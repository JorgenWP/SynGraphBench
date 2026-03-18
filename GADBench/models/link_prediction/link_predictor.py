from models.gnn import *
from sklearn.metrics import roc_auc_score, average_precision_score
import dgl
import torch
import torch.nn.functional as F


class BaseDetector(object):
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        self.neg_sampling = train_config.get('neg_sampling', 'random')
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]

        self.graph = self.data.graph.to(self.train_config['device'])
        self.train_graph = self.data.train_graph.to(self.train_config['device'])

        self.num_nodes = self.data.graph.num_nodes()
        self.edge_set = self.data.edge_set
        self.train_pos_edges = self.data.train_pos_edges.to(self.train_config['device'])
        self.val_pos_edges = self.data.val_pos_edges.to(self.train_config['device'])
        self.val_neg_edges = self.data.val_neg_edges.to(self.train_config['device'])
        self.test_pos_edges = self.data.test_pos_edges.to(self.train_config['device'])
        self.test_neg_edges = self.data.test_neg_edges.to(self.train_config['device'])

        self.best_score = -1
        self.patience_knt = 0

    def train(self):
        pass

    def eval(self, pos_scores, neg_scores):
        """Evaluate link prediction using AUROC, AUPRC, and Hits@K."""
        with torch.no_grad():
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).numpy()
            probs = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            score = {}
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            k = int(labels.sum())
            score['RecK'] = sum(labels[probs.argsort()[-k:]]) / k
        return score


class MLPDecoder(nn.Module):
    """Learnable edge scorer: MLP on Hadamard product of node embeddings."""
    def __init__(self, h_feats, dropout_rate=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, 1)
        )

    def forward(self, h, edges):
        h_hadamard = h[edges[:, 0]] * h[edges[:, 1]]
        return self.layers(h_hadamard).squeeze(-1)


class BaseGNNLinkPredictor(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['output_emb'] = True
        gnn = globals()[model_config['model']]
        self.model = gnn(**model_config).to(train_config['device'])

        h_feats = model_config.get('h_feats', 32)
        decoder = train_config.get('decoder', 'dot')
        if decoder == 'mlp':
            self.decoder = MLPDecoder(
                h_feats, model_config.get('drop_rate', 0)
            ).to(train_config['device'])
        else:
            self.decoder = None

    def score_edges(self, h, edges):
        """Score edges using dot product or MLP decoder."""
        if self.decoder is not None:
            return self.decoder(h, edges)
        return (h[edges[:, 0]] * h[edges[:, 1]]).sum(dim=-1)

    def _filter_collisions(self, neg_edges):
        """Replace negative edges that collide with real edges."""
        src, dst = neg_edges[:, 0], neg_edges[:, 1]
        N = self.num_nodes
        for attempt in range(10):
            hashes = src.long() * N + dst.long()
            collision = torch.isin(hashes, self.edge_set) | (src == dst)
            if not collision.any():
                break
            n_bad = collision.sum().item()
            dst[collision] = torch.randint(0, N, (n_bad,))
        return neg_edges

    def _sample_random_negatives(self, n):
        """Sample random node pairs as guaranteed non-edges."""
        neg_edges = torch.stack([
            torch.randint(0, self.num_nodes, (n,)),
            torch.randint(0, self.num_nodes, (n,))
        ], dim=1)
        return self._filter_collisions(neg_edges).to(self.train_config['device'])

    def _sample_hard_negatives(self, pos_edges):
        """Sample hard negatives via 2-hop random walks, guaranteed non-edges."""
        src = pos_edges[:, 0]

        # 2-step random walk on the training graph (CPU for DGL compatibility)
        walk_nodes, _ = dgl.sampling.random_walk(
            self.train_graph.cpu(), src.cpu(), metapath=[None, None])
        # walk_nodes shape: [n, 3] — columns are: start, 1-hop, 2-hop
        hard_dst = walk_nodes[:, 2]

        # Replace failed walks (-1) with random nodes
        failed = hard_dst == -1
        if failed.any():
            hard_dst[failed] = torch.randint(0, self.num_nodes, (failed.sum(),))

        neg_edges = torch.stack([src.cpu(), hard_dst], dim=1)
        return self._filter_collisions(neg_edges).to(self.train_config['device'])

    def train(self):
        params = list(self.model.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.model_config['lr'])

        test_score = None
        n_train = self.train_pos_edges.shape[0]
        for e in range(self.train_config['epochs']):
            self.model.train()
            h = self.model(self.train_graph)

            # Re-sample negative edges each epoch
            if self.neg_sampling == 'hard':
                train_neg_edges = self._sample_hard_negatives(self.train_pos_edges)
            else:
                train_neg_edges = self._sample_random_negatives(n_train)

            pos_scores = self.score_edges(h, self.train_pos_edges)
            neg_scores = self.score_edges(h, train_neg_edges)
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([
                torch.ones(pos_scores.shape[0]),
                torch.zeros(neg_scores.shape[0])
            ]).to(h.device)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.eval()
                h = self.model(self.graph)
                val_pos = torch.sigmoid(self.score_edges(h, self.val_pos_edges))
                val_neg = torch.sigmoid(self.score_edges(h, self.val_neg_edges))
            val_score = self.eval(val_pos, val_neg)

            if val_score[self.train_config['metric']] > self.best_score:
                self.best_score = val_score[self.train_config['metric']]
                self.patience_knt = 0
                with torch.no_grad():
                    test_pos = torch.sigmoid(self.score_edges(h, self.test_pos_edges))
                    test_neg = torch.sigmoid(self.score_edges(h, self.test_neg_edges))
                test_score = self.eval(test_pos, test_neg)
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, '
                      'test AUC {:.4f}, PRC {:.4f}'.format(
                          e, loss, val_score['AUROC'], val_score['AUPRC'],
                          test_score['AUROC'], test_score['AUPRC']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score
