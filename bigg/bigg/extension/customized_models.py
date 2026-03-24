# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bigg.model.tree_model import RecurTreeGen
import torch
from bigg.common.pytorch_util import glorot_uniform, MLP
import torch.nn as nn
import torch.nn.functional as F

# pylint: skip-file


class BiggWithEdgeLen(RecurTreeGen):

    def __init__(self, args):
        super().__init__(args)
        self.edgelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_encoding = MLP(1, [2 * args.embed_dim, args.embed_dim])
        self.nodelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        self.edgelen_pred = MLP(args.embed_dim, [2 * args.embed_dim, 1])
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

    # to be customized
    def embed_node_feats(self, node_feats):
        return self.nodelen_encoding(node_feats)

    def embed_edge_feats(self, edge_feats):
        return self.edgelen_encoding(edge_feats)

    def predict_node_feats(self, state, node_feats=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            node_feats: N x feat_dim or None
        Returns:
            new_state,
            likelihood of node_feats under current state,
            and, if node_feats is None, then return the prediction of node_feats
            else return the node_feats as it is
        """
        h, _ = state
        pred_node_len = self.nodelen_pred(h)
        state_update = self.embed_node_feats(pred_node_len) if node_feats is None else self.embed_node_feats(node_feats)
        new_state = self.node_state_update(state_update, state)
        if node_feats is None:
            ll = 0
            node_feats = pred_node_len
        else:
            ll = -(node_feats - pred_node_len) ** 2
            ll = torch.sum(ll)
        return new_state, ll, node_feats

    def predict_edge_feats(self, state, edge_feats=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim), the current state
            edge_feats: N x feat_dim or None
        Returns:
            likelihood of edge_feats under current state,
            and, if edge_feats is None, then return the prediction of edge_feats
            else return the edge_feats as it is
        """
        h, _ = state
        pred_edge_len = self.edgelen_pred(h)
        if edge_feats is None:
            ll = 0
            edge_feats = pred_edge_len
        else:
            ll = -(edge_feats - pred_edge_len) ** 2
            ll = torch.sum(ll) / 10.0  # need to balance the likelihood between graph structures and features
        return ll, edge_feats

class BiggWithFeatsAndLabels(RecurTreeGen):

    def __init__(self, args, feat_dim, num_classes, label_temp=1.0, noise_std=0.0, ss_prob=0.0):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.label_temp = label_temp  # sampling temperature; 1.0 = unmodified distribution
        self.noise_std = noise_std    # Gaussian noise std on hidden state during training
        self.ss_prob = ss_prob        # scheduled sampling probability (swap GT with prediction)

        # 1. Continuous feature encoders/decoders
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        self.nodefeat_pred = MLP(args.embed_dim, [2 * args.embed_dim, feat_dim])

        # 2. Discrete label encoders/decoders
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 3. Combiner to fuse both embeddings back to the expected args.embed_dim
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        self._ll_cont = 0.0
        self._ll_label = 0.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_label = 0.0

    def embed_node_feats(self, node_data):
        # node_data shape: [batch_size, feat_dim + 1]
        cont_feats = node_data[:, :self.feat_dim]
        node_labels = node_data[:, self.feat_dim].long()
        
        # Embed both separately
        embed_cont = self.nodefeat_encoding(cont_feats)
        embed_label = self.nodelabel_encoding(node_labels)
        
        # Concatenate the embeddings and project them back down to args.embed_dim
        combined_embed = torch.cat([embed_cont, embed_label], dim=-1)
        return self.combiner(combined_embed)

    def predict_node_feats(self, state, node_data=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim)
            node_data: N x (feat_dim + 1) tensor containing continuous features and labels, or None
        """
        h, c = state

        # Add Gaussian noise to hidden state during training to improve robustness
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        # Predict continuous features and class logits
        pred_cont = self.nodefeat_pred(h)
        pred_logits = self.nodelabel_pred(h)

        if node_data is None:
            # Generation mode: sample from the learned distribution
            ll = 0
            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).float()

            # Concatenate predictions back into a single tensor of shape [batch_size, feat_dim + 1]
            return_data = torch.cat([pred_cont, pred_labels], dim=-1)

            state_update = self.embed_node_feats(return_data)
        else:
            # Training mode
            target_cont = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()

            # 1. Likelihood for continuous features (Negative MSE), normalized per feature
            ll_cont = -(target_cont - pred_cont) ** 2
            ll_cont = torch.sum(ll_cont) / self.feat_dim

            # 2. Likelihood for discrete labels (Negative Cross-Entropy)
            ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')

            ll = ll_cont + ll_label

            self._ll_cont += ll_cont.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: sometimes use model's own prediction for state update
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).float()
                pred_data = torch.cat([pred_cont.detach(), ss_labels], dim=-1)
                state_update = self.embed_node_feats(pred_data)
            else:
                state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))

        return new_state, ll, return_data

class BiggWithConditionedFeats(RecurTreeGen):

    def __init__(self, args, feat_dim, num_classes, label_temp=1.0, noise_std=0.0, ss_prob=0.0):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.label_temp = label_temp  # sampling temperature; 1.0 = unmodified distribution
        self.noise_std = noise_std    # Gaussian noise std on hidden state during training
        self.ss_prob = ss_prob        # scheduled sampling probability (swap GT with prediction)

        # 1. Label encoders/decoders
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 2. Continuous feature encoders/decoders
        # Notice the input dimension to nodefeat_pred is now 2 * args.embed_dim
        # to accept both 'h' and the 'label_embedding'
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        self.nodefeat_pred = MLP(args.embed_dim * 2, [2 * args.embed_dim, feat_dim])

        # 3. Combiner for the recurrent state update
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        self._ll_cont = 0.0
        self._ll_label = 0.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_label = 0.0

    def embed_node_feats(self, node_data):
        cont_feats = node_data[:, :self.feat_dim]
        node_labels = node_data[:, self.feat_dim].long()
        
        embed_cont = self.nodefeat_encoding(cont_feats)
        embed_label = self.nodelabel_encoding(node_labels)
        
        combined_embed = torch.cat([embed_cont, embed_label], dim=-1)
        return self.combiner(combined_embed)

    def predict_node_feats(self, state, node_data=None):
        h, c = state

        # Add Gaussian noise to hidden state during training to improve robustness
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        # Step 1: Predict class logits from the hidden state 'h'
        pred_logits = self.nodelabel_pred(h)

        if node_data is None:
            # --- Generation Mode ---
            ll = 0

            # Sample label from the learned distribution (with optional temperature)
            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Step 2: Embed the sampled label
            label_embed = self.nodelabel_encoding(pred_labels)

            # Step 3 & 4: Condition continuous features on both 'h' and 'label_embed'
            h_conditioned = torch.cat([h, label_embed], dim=-1)
            pred_cont = self.nodefeat_pred(h_conditioned)

            # Format output
            return_data = torch.cat([pred_cont, pred_labels.unsqueeze(-1).float()], dim=-1)
            state_update = self.embed_node_feats(return_data)

        else:
            # --- Training Mode ---
            target_cont = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()

            # Step 2: Embed the GROUND TRUTH label (Teacher Forcing)
            label_embed = self.nodelabel_encoding(target_labels)

            # Step 3 & 4: Condition continuous features on both 'h' and true 'label_embed'
            h_conditioned = torch.cat([h, label_embed], dim=-1)
            pred_cont = self.nodefeat_pred(h_conditioned)

            # Compute losses
            # Normalize ll_cont per feature so it's on the same scale as ll_label
            ll_cont = -(target_cont - pred_cont) ** 2
            ll_cont = torch.sum(ll_cont) / self.feat_dim

            ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')

            ll = ll_cont + ll_label

            self._ll_cont += ll_cont.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: sometimes use model's own prediction for state update
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).squeeze(-1)
                pred_data = torch.cat([pred_cont.detach(), ss_labels.unsqueeze(-1).float()], dim=-1)
                state_update = self.embed_node_feats(pred_data)
            else:
                state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))

        return new_state, ll, return_data