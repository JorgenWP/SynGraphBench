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

    def __init__(self, args, feat_dim, num_classes, label_temp=1.0, noise_std=0.0, ss_prob=0.0,
                 hetero_feat=False, logvar_floor=-4.0, binary_feat=False, binary_idx=None):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.label_temp = label_temp  # sampling temperature; 1.0 = unmodified distribution
        self.noise_std = noise_std    # Gaussian noise std on hidden state during training
        self.ss_prob = ss_prob        # scheduled sampling probability (swap GT with prediction)
        self.hetero_feat = hetero_feat  # predict mean + log-variance for continuous features
        self.logvar_floor = logvar_floor
        self.binary_feat = binary_feat

        # Binary / continuous feature index split
        if binary_feat and binary_idx:
            self.binary_idx = sorted(binary_idx)
            self.cont_idx = sorted(set(range(feat_dim)) - set(self.binary_idx))
        else:
            self.binary_idx = []
            self.cont_idx = list(range(feat_dim))
        self.cont_feat_dim = len(self.cont_idx)
        self.bin_feat_dim = len(self.binary_idx)

        # 1. Feature encoding (all features together)
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])

        # 2. Continuous feature decoder
        cont_out_dim = 2 * self.cont_feat_dim if hetero_feat else self.cont_feat_dim
        self.nodefeat_pred = MLP(args.embed_dim, [2 * args.embed_dim, cont_out_dim])

        # 3. Binary feature decoder (logits for BCE)
        if self.bin_feat_dim > 0:
            self.binfeat_pred = MLP(args.embed_dim, [2 * args.embed_dim, self.bin_feat_dim])

        # 4. Discrete label encoders/decoders
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 5. Combiner to fuse both embeddings back to the expected args.embed_dim
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0

        # Loss weights (set by pipeline for dynamic normalization)
        self.w_cont = 1.0
        self.w_bin = 1.0
        self.w_label = 1.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0

    def _assemble_features(self, cont_vals, bin_vals):
        """Reassemble full feat_dim vector from continuous and binary parts."""
        batch = cont_vals.shape[0]
        full = torch.empty(batch, self.feat_dim, device=cont_vals.device)
        if self.cont_feat_dim > 0:
            full[:, self.cont_idx] = cont_vals
        if self.bin_feat_dim > 0:
            full[:, self.binary_idx] = bin_vals
        return full

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

    def predict_node_feats(self, state, node_data=None, label_mask=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim)
            node_data: N x (feat_dim + 1) tensor containing continuous features and labels, or None
            label_mask: N boolean tensor — True = include in label loss (None = all)
        """
        h, c = state

        # Add Gaussian noise to hidden state during training to improve robustness
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        # Predict continuous features and class logits
        raw_cont = self.nodefeat_pred(h)
        pred_logits = self.nodelabel_pred(h)

        if self.hetero_feat:
            pred_cont = raw_cont[:, :self.cont_feat_dim]
            log_var = torch.clamp(raw_cont[:, self.cont_feat_dim:], self.logvar_floor, 2.0)
        else:
            pred_cont = raw_cont

        # Predict binary features (logits)
        bin_logits = self.binfeat_pred(h) if self.bin_feat_dim > 0 else None

        if node_data is None:
            # Generation mode: sample from the learned distribution
            ll = 0
            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).float()

            if self.hetero_feat:
                std = torch.exp(0.5 * log_var)
                sampled_cont = pred_cont + std * torch.randn_like(pred_cont)
            else:
                sampled_cont = pred_cont

            # Binary: Bernoulli sample from sigmoid(logits)
            if bin_logits is not None:
                sampled_bin = torch.bernoulli(torch.sigmoid(bin_logits))
            else:
                sampled_bin = torch.empty(h.shape[0], 0, device=h.device)

            all_feats = self._assemble_features(sampled_cont, sampled_bin)
            return_data = torch.cat([all_feats, pred_labels], dim=-1)
            state_update = self.embed_node_feats(return_data)
        else:
            # Training mode
            target_all = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()

            # Split targets by feature type
            target_cont = target_all[:, self.cont_idx] if self.cont_feat_dim > 0 else None
            target_bin = target_all[:, self.binary_idx] if self.bin_feat_dim > 0 else None

            # 1. Likelihood for continuous features
            if target_cont is not None and self.cont_feat_dim > 0:
                if self.hetero_feat:
                    ll_cont = -0.5 * (log_var + (target_cont - pred_cont) ** 2 / torch.exp(log_var))
                else:
                    ll_cont = -(target_cont - pred_cont) ** 2
                ll_cont = torch.sum(ll_cont) / self.cont_feat_dim
            else:
                ll_cont = torch.tensor(0.0, device=h.device)

            # 2. Likelihood for binary features (negative BCE)
            if target_bin is not None and self.bin_feat_dim > 0:
                ll_bin = -F.binary_cross_entropy_with_logits(bin_logits, target_bin, reduction='sum')
                ll_bin = ll_bin / self.bin_feat_dim
            else:
                ll_bin = torch.tensor(0.0, device=h.device)

            # 3. Likelihood for discrete labels (Negative Cross-Entropy)
            if label_mask is not None and not label_mask.all():
                ll_label = -F.cross_entropy(pred_logits[label_mask], target_labels[label_mask], reduction='sum')
            else:
                ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')

            ll = self.w_cont * ll_cont + self.w_bin * ll_bin + self.w_label * ll_label

            self._ll_cont += ll_cont.item()
            self._ll_bin += ll_bin.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: sometimes use model's own prediction for state update
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).float()
                ss_feats = self._assemble_features(
                    pred_cont.detach(),
                    torch.sigmoid(bin_logits).detach() if bin_logits is not None
                    else torch.empty(h.shape[0], 0, device=h.device)
                )
                pred_data = torch.cat([ss_feats, ss_labels], dim=-1)
                state_update = self.embed_node_feats(pred_data)
            else:
                state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))

        return new_state, ll, return_data

class BiggWithConditionedFeats(RecurTreeGen):

    def __init__(self, args, feat_dim, num_classes, label_temp=1.0, noise_std=0.0, ss_prob=0.0,
                 hetero_feat=False, logvar_floor=-4.0, binary_feat=False, binary_idx=None):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.label_temp = label_temp
        self.noise_std = noise_std
        self.ss_prob = ss_prob
        self.hetero_feat = hetero_feat
        self.logvar_floor = logvar_floor
        self.binary_feat = binary_feat

        # Binary / continuous feature index split
        if binary_feat and binary_idx:
            self.binary_idx = sorted(binary_idx)
            self.cont_idx = sorted(set(range(feat_dim)) - set(self.binary_idx))
        else:
            self.binary_idx = []
            self.cont_idx = list(range(feat_dim))
        self.cont_feat_dim = len(self.cont_idx)
        self.bin_feat_dim = len(self.binary_idx)

        # 1. Label encoders/decoders
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 2. Continuous feature encoders/decoders
        # Input to nodefeat_pred is 2*embed_dim to accept both 'h' and 'label_embedding'
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        cont_out_dim = 2 * self.cont_feat_dim if hetero_feat else self.cont_feat_dim
        self.nodefeat_pred = MLP(args.embed_dim * 2, [2 * args.embed_dim, cont_out_dim])

        # 3. Binary feature decoder (logits for BCE), also conditioned on label
        if self.bin_feat_dim > 0:
            self.binfeat_pred = MLP(args.embed_dim * 2, [2 * args.embed_dim, self.bin_feat_dim])

        # 4. Combiner for the recurrent state update
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0

        # Loss weights (set by pipeline for dynamic normalization)
        self.w_cont = 1.0
        self.w_bin = 1.0
        self.w_label = 1.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0

    def _assemble_features(self, cont_vals, bin_vals):
        """Reassemble full feat_dim vector from continuous and binary parts."""
        batch = cont_vals.shape[0]
        full = torch.empty(batch, self.feat_dim, device=cont_vals.device)
        if self.cont_feat_dim > 0:
            full[:, self.cont_idx] = cont_vals
        if self.bin_feat_dim > 0:
            full[:, self.binary_idx] = bin_vals
        return full

    def embed_node_feats(self, node_data):
        cont_feats = node_data[:, :self.feat_dim]
        node_labels = node_data[:, self.feat_dim].long()

        embed_cont = self.nodefeat_encoding(cont_feats)
        embed_label = self.nodelabel_encoding(node_labels)

        combined_embed = torch.cat([embed_cont, embed_label], dim=-1)
        return self.combiner(combined_embed)

    def predict_node_feats(self, state, node_data=None, label_mask=None):
        """
        Args:
            state: tuple of (h=N x embed_dim, c=N x embed_dim)
            node_data: N x (feat_dim + 1) tensor containing continuous features and labels, or None
            label_mask: N boolean tensor — True = include in label loss (None = all)
        """
        h, c = state

        # Add Gaussian noise to hidden state during training
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        # Step 1: Predict class logits from the hidden state 'h'
        pred_logits = self.nodelabel_pred(h)

        if node_data is None:
            # --- Generation Mode ---
            ll = 0

            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Step 2: Embed the sampled label
            label_embed = self.nodelabel_encoding(pred_labels)

            # Step 3: Condition features on both 'h' and 'label_embed'
            h_conditioned = torch.cat([h, label_embed], dim=-1)
            raw_cont = self.nodefeat_pred(h_conditioned)

            if self.hetero_feat:
                pred_cont = raw_cont[:, :self.cont_feat_dim]
                log_var = torch.clamp(raw_cont[:, self.cont_feat_dim:], self.logvar_floor, 2.0)
                std = torch.exp(0.5 * log_var)
                sampled_cont = pred_cont + std * torch.randn_like(pred_cont)
            else:
                sampled_cont = raw_cont

            # Binary: Bernoulli sample from sigmoid(logits)
            if self.bin_feat_dim > 0:
                bin_logits = self.binfeat_pred(h_conditioned)
                sampled_bin = torch.bernoulli(torch.sigmoid(bin_logits))
            else:
                sampled_bin = torch.empty(h.shape[0], 0, device=h.device)

            all_feats = self._assemble_features(sampled_cont, sampled_bin)
            return_data = torch.cat([all_feats, pred_labels.unsqueeze(-1).float()], dim=-1)
            state_update = self.embed_node_feats(return_data)

        else:
            # --- Training Mode ---
            target_all = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()

            # Step 2: Embed the GROUND TRUTH label (Teacher Forcing)
            label_embed = self.nodelabel_encoding(target_labels)

            # Step 3: Condition features on both 'h' and true 'label_embed'
            h_conditioned = torch.cat([h, label_embed], dim=-1)
            raw_cont = self.nodefeat_pred(h_conditioned)

            if self.hetero_feat:
                pred_cont = raw_cont[:, :self.cont_feat_dim]
                log_var = torch.clamp(raw_cont[:, self.cont_feat_dim:], self.logvar_floor, 2.0)
            else:
                pred_cont = raw_cont

            # Split targets by feature type
            target_cont = target_all[:, self.cont_idx] if self.cont_feat_dim > 0 else None
            target_bin = target_all[:, self.binary_idx] if self.bin_feat_dim > 0 else None

            # 1. Likelihood for continuous features
            if target_cont is not None and self.cont_feat_dim > 0:
                if self.hetero_feat:
                    ll_cont = -0.5 * (log_var + (target_cont - pred_cont) ** 2 / torch.exp(log_var))
                else:
                    ll_cont = -(target_cont - pred_cont) ** 2
                ll_cont = torch.sum(ll_cont) / self.cont_feat_dim
            else:
                ll_cont = torch.tensor(0.0, device=h.device)

            # 2. Likelihood for binary features (negative BCE)
            if target_bin is not None and self.bin_feat_dim > 0:
                bin_logits = self.binfeat_pred(h_conditioned)
                ll_bin = -F.binary_cross_entropy_with_logits(bin_logits, target_bin, reduction='sum')
                ll_bin = ll_bin / self.bin_feat_dim
            else:
                ll_bin = torch.tensor(0.0, device=h.device)

            # 3. Likelihood for discrete labels (Negative Cross-Entropy)
            if label_mask is not None and not label_mask.all():
                ll_label = -F.cross_entropy(pred_logits[label_mask], target_labels[label_mask], reduction='sum')
            else:
                ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')

            ll = self.w_cont * ll_cont + self.w_bin * ll_bin + self.w_label * ll_label

            self._ll_cont += ll_cont.item()
            self._ll_bin += ll_bin.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: sometimes use model's own prediction for state update
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).squeeze(-1)
                ss_feats = self._assemble_features(
                    pred_cont.detach(),
                    torch.sigmoid(self.binfeat_pred(h_conditioned)).detach() if self.bin_feat_dim > 0
                    else torch.empty(h.shape[0], 0, device=h.device)
                )
                pred_data = torch.cat([ss_feats, ss_labels.unsqueeze(-1).float()], dim=-1)
                state_update = self.embed_node_feats(pred_data)
            else:
                state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))

        return new_state, ll, return_data