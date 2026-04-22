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

import math

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
                 hetero_feat=False, logvar_floor=-4.0, binary_feat=False, binary_idx=None,
                 vae_feat=False, vae_dim=16, kl_weight=1.0):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.label_temp = label_temp  # sampling temperature; 1.0 = unmodified distribution
        self.noise_std = noise_std    # Gaussian noise std on hidden state during training
        self.ss_prob = ss_prob        # scheduled sampling probability (swap GT with prediction)
        self.hetero_feat = hetero_feat  # predict mean + log-variance for continuous features
        self.logvar_floor = logvar_floor
        self.binary_feat = binary_feat
        self.vae_feat = vae_feat
        self.vae_dim = vae_dim if vae_feat else 0
        self.kl_weight = kl_weight

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

        # 2. Continuous feature decoder (takes h optionally concatenated with z)
        feat_in_dim = args.embed_dim + self.vae_dim
        cont_out_dim = 2 * self.cont_feat_dim if hetero_feat else self.cont_feat_dim
        self.nodefeat_pred = MLP(feat_in_dim, [2 * args.embed_dim, cont_out_dim])

        # 3. Binary feature decoder (logits for BCE)
        if self.bin_feat_dim > 0:
            self.binfeat_pred = MLP(feat_in_dim, [2 * args.embed_dim, self.bin_feat_dim])

        # 4. Discrete label encoders/decoders (label head unchanged — predicted from h only)
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 5. Combiner to fuse both embeddings back to the expected args.embed_dim
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        # 6. VAE encoder (training only; constructed only when enabled)
        if self.vae_feat:
            self.vae_encoder = MLP(args.embed_dim + feat_dim,
                                   [2 * args.embed_dim, 2 * self.vae_dim])

        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

        # Loss weights (set by pipeline for dynamic normalization)
        self.w_cont = 1.0
        self.w_bin = 1.0
        self.w_label = 1.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

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

        # Class logits are predicted from h only (label head is not VAE-conditioned)
        pred_logits = self.nodelabel_pred(h)

        # VAE latent: posterior q(z|h,x) at training time, prior N(0,I) at generation time.
        mu = log_var_z = None
        if self.vae_feat:
            if node_data is not None:
                target_all = node_data[:, :self.feat_dim]
                enc_out = self.vae_encoder(torch.cat([h, target_all], dim=-1))
                mu = enc_out[:, :self.vae_dim]
                log_var_z = torch.clamp(enc_out[:, self.vae_dim:], self.logvar_floor, 2.0)
                z = mu + torch.exp(0.5 * log_var_z) * torch.randn_like(mu)
            else:
                z = torch.randn(h.shape[0], self.vae_dim, device=h.device)
            h_feat_in = torch.cat([h, z], dim=-1)
        else:
            h_feat_in = h

        # Predict continuous features
        raw_cont = self.nodefeat_pred(h_feat_in)
        if self.hetero_feat:
            pred_cont = raw_cont[:, :self.cont_feat_dim]
            log_var = torch.clamp(raw_cont[:, self.cont_feat_dim:], self.logvar_floor, 2.0)
        else:
            pred_cont = raw_cont

        # Predict binary features (logits)
        bin_logits = self.binfeat_pred(h_feat_in) if self.bin_feat_dim > 0 else None

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

            # 4. KL(q(z|h,x) || N(0,I)) — subtract from ll since we maximize ll
            if self.vae_feat:
                kl = 0.5 * torch.sum(torch.exp(log_var_z) + mu ** 2 - 1.0 - log_var_z, dim=-1)
                kl_total = kl.sum() / self.vae_dim
                ll = ll - self.kl_weight * kl_total
                self._kl += kl_total.item()

            self._ll_cont += ll_cont.item()
            self._ll_bin += ll_bin.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: sometimes use model's own prediction for state update.
            # With VAE on, regenerate features from a fresh prior z (generation-time distribution).
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).float()
                    if self.vae_feat:
                        z_prior = torch.randn(h.shape[0], self.vae_dim, device=h.device)
                        h_ss_in = torch.cat([h, z_prior], dim=-1)
                        raw_cont_ss = self.nodefeat_pred(h_ss_in)
                        pred_cont_ss = raw_cont_ss[:, :self.cont_feat_dim] if self.hetero_feat else raw_cont_ss
                        bin_logits_ss = self.binfeat_pred(h_ss_in) if self.bin_feat_dim > 0 else None
                    else:
                        pred_cont_ss = pred_cont
                        bin_logits_ss = bin_logits
                ss_feats = self._assemble_features(
                    pred_cont_ss.detach(),
                    torch.sigmoid(bin_logits_ss).detach() if bin_logits_ss is not None
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
                 hetero_feat=False, logvar_floor=-4.0, binary_feat=False, binary_idx=None,
                 vae_feat=False, vae_dim=16, kl_weight=1.0):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.label_temp = label_temp
        self.noise_std = noise_std
        self.ss_prob = ss_prob
        self.hetero_feat = hetero_feat
        self.logvar_floor = logvar_floor
        self.binary_feat = binary_feat
        self.vae_feat = vae_feat
        self.vae_dim = vae_dim if vae_feat else 0
        self.kl_weight = kl_weight

        # Binary / continuous feature index split
        if binary_feat and binary_idx:
            self.binary_idx = sorted(binary_idx)
            self.cont_idx = sorted(set(range(feat_dim)) - set(self.binary_idx))
        else:
            self.binary_idx = []
            self.cont_idx = list(range(feat_dim))
        self.cont_feat_dim = len(self.cont_idx)
        self.bin_feat_dim = len(self.binary_idx)

        # 1. Label encoders/decoders (label head predicts from h only — not VAE-conditioned)
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 2. Continuous feature encoders/decoders.
        # Decoder input is [h, label_embed, (z)]: 2*embed_dim + vae_dim
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        feat_in_dim = args.embed_dim * 2 + self.vae_dim
        cont_out_dim = 2 * self.cont_feat_dim if hetero_feat else self.cont_feat_dim
        self.nodefeat_pred = MLP(feat_in_dim, [2 * args.embed_dim, cont_out_dim])

        # 3. Binary feature decoder (logits for BCE), also conditioned on label (and z)
        if self.bin_feat_dim > 0:
            self.binfeat_pred = MLP(feat_in_dim, [2 * args.embed_dim, self.bin_feat_dim])

        # 4. Combiner for the recurrent state update
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        # 5. VAE encoder (training only; constructed only when enabled)
        if self.vae_feat:
            self.vae_encoder = MLP(args.embed_dim + feat_dim,
                                   [2 * args.embed_dim, 2 * self.vae_dim])

        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

        # Loss weights (set by pipeline for dynamic normalization)
        self.w_cont = 1.0
        self.w_bin = 1.0
        self.w_label = 1.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

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

        # Step 1: Predict class logits from h (label head is not VAE-conditioned)
        pred_logits = self.nodelabel_pred(h)

        # VAE latent: posterior q(z|h,x) during training, prior N(0,I) at generation.
        mu = log_var_z = None
        if self.vae_feat:
            if node_data is not None:
                target_all_for_enc = node_data[:, :self.feat_dim]
                enc_out = self.vae_encoder(torch.cat([h, target_all_for_enc], dim=-1))
                mu = enc_out[:, :self.vae_dim]
                log_var_z = torch.clamp(enc_out[:, self.vae_dim:], self.logvar_floor, 2.0)
                z = mu + torch.exp(0.5 * log_var_z) * torch.randn_like(mu)
            else:
                z = torch.randn(h.shape[0], self.vae_dim, device=h.device)

        if node_data is None:
            # --- Generation Mode ---
            ll = 0

            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Step 2: Embed the sampled label
            label_embed = self.nodelabel_encoding(pred_labels)

            # Step 3: Condition features on h, label_embed, and (if enabled) z
            if self.vae_feat:
                h_conditioned = torch.cat([h, label_embed, z], dim=-1)
            else:
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

            # Step 3: Condition features on h, true label_embed, and (if enabled) z
            if self.vae_feat:
                h_conditioned = torch.cat([h, label_embed, z], dim=-1)
            else:
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

            # 4. KL(q(z|h,x) || N(0,I)) — subtract since we maximize ll
            if self.vae_feat:
                kl = 0.5 * torch.sum(torch.exp(log_var_z) + mu ** 2 - 1.0 - log_var_z, dim=-1)
                kl_total = kl.sum() / self.vae_dim
                ll = ll - self.kl_weight * kl_total
                self._kl += kl_total.item()

            self._ll_cont += ll_cont.item()
            self._ll_bin += ll_bin.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: sometimes use model's own prediction for state update.
            # With VAE on, regenerate features from a fresh prior z (generation-time distribution).
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).squeeze(-1)
                    if self.vae_feat:
                        z_prior = torch.randn(h.shape[0], self.vae_dim, device=h.device)
                        h_ss_in = torch.cat([h, label_embed, z_prior], dim=-1)
                        raw_cont_ss = self.nodefeat_pred(h_ss_in)
                        pred_cont_ss = raw_cont_ss[:, :self.cont_feat_dim] if self.hetero_feat else raw_cont_ss
                        bin_logits_ss = self.binfeat_pred(h_ss_in) if self.bin_feat_dim > 0 else None
                    else:
                        pred_cont_ss = pred_cont
                        bin_logits_ss = self.binfeat_pred(h_conditioned) if self.bin_feat_dim > 0 else None
                ss_feats = self._assemble_features(
                    pred_cont_ss.detach(),
                    torch.sigmoid(bin_logits_ss).detach() if bin_logits_ss is not None
                    else torch.empty(h.shape[0], 0, device=h.device)
                )
                pred_data = torch.cat([ss_feats, ss_labels.unsqueeze(-1).float()], dim=-1)
                state_update = self.embed_node_feats(pred_data)
            else:
                state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))

        return new_state, ll, return_data


class BiggWithARCatFeats(RecurTreeGen):
    """BiGG with autoregressive categorical feature prediction.

    Continuous feature columns are binned per-feature with fixed (pre-fit) quantile
    edges. At each node the label is predicted from h, then continuous features are
    generated autoregressively across feature positions, conditioned on
    [h, label_embed, feat_pos_embed, prev_bin_embed].

    Training: parallel over feature positions (teacher-forced previous bins).
    Generation: sequential loop over feature positions.

    Binary columns keep the Bernoulli/BCE head from the baseline models.

    Continuous loss is value-space Gaussian soft-label cross-entropy, so the
    penalty reflects feature-space distance (not bin-index distance) and handles
    mass-point duplicates naturally.
    """

    def __init__(self, args, feat_dim, num_classes, n_bins, bin_edges, bin_centers,
                 bin_sigma, label_temp=1.0, noise_std=0.0,
                 binary_feat=False, binary_idx=None):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.n_bins = n_bins
        self.label_temp = label_temp
        self.noise_std = noise_std

        # Binary / continuous split (same convention as sister classes)
        if binary_feat and binary_idx:
            self.binary_idx = sorted(binary_idx)
            self.cont_idx = sorted(set(range(feat_dim)) - set(self.binary_idx))
        else:
            self.binary_idx = []
            self.cont_idx = list(range(feat_dim))
        self.cont_feat_dim = len(self.cont_idx)
        self.bin_feat_dim = len(self.binary_idx)

        # Bin buffers come from a fitted FeatureBinner; saved with state_dict and
        # move with .to(device). Shapes:
        #   bin_edges:   (cont_feat_dim, n_bins - 1)
        #   bin_centers: (cont_feat_dim, n_bins)
        if self.cont_feat_dim > 0:
            assert bin_edges.shape == (self.cont_feat_dim, n_bins - 1), \
                f"bin_edges shape {bin_edges.shape} != ({self.cont_feat_dim}, {n_bins - 1})"
            assert bin_centers.shape == (self.cont_feat_dim, n_bins), \
                f"bin_centers shape {bin_centers.shape} != ({self.cont_feat_dim}, {n_bins})"
            self.register_buffer('bin_edges', bin_edges.float())
            self.register_buffer('bin_centers', bin_centers.float())

            # Per-feature sigma: accept scalar (broadcast to F) or (F,) tensor.
            if torch.is_tensor(bin_sigma):
                sigma_tensor = bin_sigma.float()
                if sigma_tensor.dim() == 0:
                    sigma_tensor = sigma_tensor.expand(self.cont_feat_dim).clone()
                assert sigma_tensor.shape == (self.cont_feat_dim,), \
                    f"bin_sigma shape {sigma_tensor.shape} != ({self.cont_feat_dim},)"
            else:
                sigma_tensor = torch.full((self.cont_feat_dim,), float(bin_sigma))
            self.register_buffer('bin_sigma', sigma_tensor)

        # 1. Label head (from h)
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # 2. AR head: shared MLP over (h, label_embed, feat_pos_embed, prev_bin_embed).
        # prev_bin vocabulary is n_bins + 1 to reserve <BOS> at feature position 0.
        if self.cont_feat_dim > 0:
            self.feat_pos_embed = nn.Embedding(self.cont_feat_dim, args.embed_dim)
            self.prev_bin_embed = nn.Embedding(n_bins + 1, args.embed_dim)
            ar_in_dim = args.embed_dim * 4
            self.ar_head = MLP(ar_in_dim, [2 * args.embed_dim, n_bins])

        # 3. Binary head (conditioned on h + label_embed only)
        if self.bin_feat_dim > 0:
            self.binfeat_pred = MLP(args.embed_dim * 2, [2 * args.embed_dim, self.bin_feat_dim])

        # 4. State update path (matches sister classes)
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        # Loss tracking (_kl kept at 0 for pipeline compat)
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

        self.w_cont = 1.0
        self.w_bin = 1.0
        self.w_label = 1.0

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

    def _assemble_features(self, cont_vals, bin_vals):
        if cont_vals.numel() > 0:
            batch, device = cont_vals.shape[0], cont_vals.device
        else:
            batch, device = bin_vals.shape[0], bin_vals.device
        full = torch.empty(batch, self.feat_dim, device=device)
        if self.cont_feat_dim > 0:
            full[:, self.cont_idx] = cont_vals
        if self.bin_feat_dim > 0:
            full[:, self.binary_idx] = bin_vals
        return full

    def _values_to_bins(self, values):
        """Map continuous values to bin indices using fitted edges.

        values: (N, cont_feat_dim) in the same space as fitted edges.
        returns: (N, cont_feat_dim) long tensor of bin indices in [0, n_bins).
        """
        # searchsorted needs boundaries and values with matching shape except
        # on the last axis. Work in (F, N) orientation, then transpose back.
        idx = torch.searchsorted(self.bin_edges, values.T.contiguous())
        return idx.clamp(max=self.n_bins - 1).T

    def _value_space_soft_targets(self, values):
        """Gaussian soft-label distribution over bins, in value space.

        values: (N, cont_feat_dim)
        returns: (N, cont_feat_dim, n_bins) probability distribution centered at
        the true value with std bin_sigma (in feature-value units).
        """
        centers = self.bin_centers.unsqueeze(0)           # (1, F, B)
        vals = values.unsqueeze(-1)                       # (N, F, 1)
        sigma = self.bin_sigma.view(1, -1, 1)             # (1, F, 1)
        log_kernel = -((centers - vals) ** 2) / (2 * sigma ** 2)
        return F.softmax(log_kernel, dim=-1)

    def embed_node_feats(self, node_data):
        cont_feats = node_data[:, :self.feat_dim]
        node_labels = node_data[:, self.feat_dim].long()
        embed_cont = self.nodefeat_encoding(cont_feats)
        embed_label = self.nodelabel_encoding(node_labels)
        combined_embed = torch.cat([embed_cont, embed_label], dim=-1)
        return self.combiner(combined_embed)

    def predict_node_feats(self, state, node_data=None, label_mask=None):
        h, c = state
        N, D = h.shape

        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        pred_logits = self.nodelabel_pred(h)

        if node_data is None:
            # --- Generation Mode ---
            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
            label_embed = self.nodelabel_encoding(pred_labels)

            if self.cont_feat_dim > 0:
                sampled_cont = torch.empty(N, self.cont_feat_dim, device=h.device)
                prev_bin = torch.full((N,), self.n_bins, dtype=torch.long, device=h.device)
                for f in range(self.cont_feat_dim):
                    feat_pos = torch.full((N,), f, dtype=torch.long, device=h.device)
                    ar_in = torch.cat([h, label_embed,
                                       self.feat_pos_embed(feat_pos),
                                       self.prev_bin_embed(prev_bin)], dim=-1)
                    logits = self.ar_head(ar_in)
                    bin_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
                    sampled_cont[:, f] = self.bin_centers[f, bin_idx]
                    prev_bin = bin_idx
            else:
                sampled_cont = torch.empty(N, 0, device=h.device)

            if self.bin_feat_dim > 0:
                bin_logits = self.binfeat_pred(torch.cat([h, label_embed], dim=-1))
                sampled_bin = torch.bernoulli(torch.sigmoid(bin_logits))
            else:
                sampled_bin = torch.empty(N, 0, device=h.device)

            all_feats = self._assemble_features(sampled_cont, sampled_bin)
            return_data = torch.cat([all_feats, pred_labels.unsqueeze(-1).float()], dim=-1)
            state_update = self.embed_node_feats(return_data)
            ll = 0

        else:
            # --- Training Mode ---
            target_all = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()
            label_embed = self.nodelabel_encoding(target_labels)

            target_cont = target_all[:, self.cont_idx] if self.cont_feat_dim > 0 else None
            target_bin = target_all[:, self.binary_idx] if self.bin_feat_dim > 0 else None

            # 1. Continuous AR loss (batched across feature positions via teacher forcing)
            if target_cont is not None and self.cont_feat_dim > 0:
                F_cont = self.cont_feat_dim
                true_bins = self._values_to_bins(target_cont)           # (N, F_cont)
                bos = torch.full((N, 1), self.n_bins, dtype=torch.long, device=h.device)
                prev_bins = torch.cat([bos, true_bins[:, :-1]], dim=1)  # (N, F_cont)

                h_exp = h.unsqueeze(1).expand(-1, F_cont, -1).reshape(N * F_cont, D)
                label_exp = label_embed.unsqueeze(1).expand(-1, F_cont, -1).reshape(N * F_cont, D)
                feat_pos = torch.arange(F_cont, device=h.device).unsqueeze(0).expand(N, -1).reshape(-1)
                pos_embed = self.feat_pos_embed(feat_pos)
                prev_embed = self.prev_bin_embed(prev_bins.reshape(-1))

                ar_in = torch.cat([h_exp, label_exp, pos_embed, prev_embed], dim=-1)
                logits = self.ar_head(ar_in).reshape(N, F_cont, self.n_bins)

                soft_targets = self._value_space_soft_targets(target_cont)  # (N, F_cont, n_bins)
                log_probs = F.log_softmax(logits, dim=-1)
                ll_cont = (soft_targets * log_probs).sum() / F_cont
            else:
                ll_cont = torch.tensor(0.0, device=h.device)

            # 2. Binary BCE
            if target_bin is not None and self.bin_feat_dim > 0:
                bin_logits = self.binfeat_pred(torch.cat([h, label_embed], dim=-1))
                ll_bin = -F.binary_cross_entropy_with_logits(bin_logits, target_bin, reduction='sum')
                ll_bin = ll_bin / self.bin_feat_dim
            else:
                ll_bin = torch.tensor(0.0, device=h.device)

            # 3. Label cross-entropy
            if label_mask is not None and not label_mask.all():
                ll_label = -F.cross_entropy(pred_logits[label_mask], target_labels[label_mask], reduction='sum')
            else:
                ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')

            ll = self.w_cont * ll_cont + self.w_bin * ll_bin + self.w_label * ll_label

            self._ll_cont += ll_cont.item()
            self._ll_bin += ll_bin.item()
            self._ll_label += ll_label.item()

            state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))
        return new_state, ll, return_data


class BiggWithMDNFeats(RecurTreeGen):
    """BiGG with a Mixture Density Network continuous feature head.

    Per-feature K-component Gaussian mixture conditioned on [h, label_embed].
    For each node and each continuous feature column, the head emits K
    (logit_pi, mu, log_sigma) triples; training loss is the negative
    log-likelihood of the true value under the mixture, and generation samples
    a component index then a Gaussian.

    Binary columns keep the Bernoulli/BCE head. No quantile bins, no AR chain,
    no latent — just a direct mixture. Works uniformly across embedding / mass-
    point / near-binary regimes because components collapse onto modes as
    needed.
    """

    def __init__(self, args, feat_dim, num_classes, n_components,
                 label_temp=1.0, noise_std=0.0, logsigma_floor=-4.0,
                 binary_feat=False, binary_idx=None):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.n_components = n_components
        self.label_temp = label_temp
        self.noise_std = noise_std
        self.logsigma_floor = logsigma_floor

        if binary_feat and binary_idx:
            self.binary_idx = sorted(binary_idx)
            self.cont_idx = sorted(set(range(feat_dim)) - set(self.binary_idx))
        else:
            self.binary_idx = []
            self.cont_idx = list(range(feat_dim))
        self.cont_feat_dim = len(self.cont_idx)
        self.bin_feat_dim = len(self.binary_idx)

        # Label head (from h)
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])

        # MDN head: per-node emits F * K * 3 parameters (pi_logit, mu, log_sigma).
        # Input is [h, label_embed]: 2 * embed_dim.
        if self.cont_feat_dim > 0:
            mdn_out_dim = self.cont_feat_dim * n_components * 3
            self.mdn_head = MLP(args.embed_dim * 2, [2 * args.embed_dim, mdn_out_dim])

        # Binary head (same conditioning as MDN)
        if self.bin_feat_dim > 0:
            self.binfeat_pred = MLP(args.embed_dim * 2, [2 * args.embed_dim, self.bin_feat_dim])

        # State update path
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

        self.w_cont = 1.0
        self.w_bin = 1.0
        self.w_label = 1.0

        self.ss_prob = 0.0  # scheduled sampling probability (set per-epoch by pipeline)

    def reset_loss_trackers(self):
        self._ll_cont = 0.0
        self._ll_bin = 0.0
        self._ll_label = 0.0
        self._kl = 0.0

    def _assemble_features(self, cont_vals, bin_vals):
        if cont_vals.numel() > 0:
            batch, device = cont_vals.shape[0], cont_vals.device
        else:
            batch, device = bin_vals.shape[0], bin_vals.device
        full = torch.empty(batch, self.feat_dim, device=device)
        if self.cont_feat_dim > 0:
            full[:, self.cont_idx] = cont_vals
        if self.bin_feat_dim > 0:
            full[:, self.binary_idx] = bin_vals
        return full

    def _sample_mdn(self, h_cond):
        """Sample one continuous value per (node, feature) from the MDN."""
        N = h_cond.shape[0]
        log_pi, mu, log_sigma = self._mdn_params(h_cond)
        pi = log_pi.exp()
        flat_pi = pi.reshape(N * self.cont_feat_dim, self.n_components)
        comp_idx = torch.multinomial(flat_pi, num_samples=1).squeeze(-1)
        comp_idx = comp_idx.view(N, self.cont_feat_dim)
        gather_idx = comp_idx.unsqueeze(-1)
        chosen_mu = mu.gather(-1, gather_idx).squeeze(-1)
        chosen_log_sigma = log_sigma.gather(-1, gather_idx).squeeze(-1)
        return chosen_mu + chosen_log_sigma.exp() * torch.randn_like(chosen_mu)

    def _mdn_params(self, h_cond):
        """Split MDN head output into (log_pi, mu, log_sigma), each (N, F, K)."""
        N = h_cond.shape[0]
        raw = self.mdn_head(h_cond).view(N, self.cont_feat_dim, self.n_components, 3)
        pi_logits = raw[..., 0]
        mu = raw[..., 1]
        log_sigma = torch.clamp(raw[..., 2], min=self.logsigma_floor, max=2.0)
        log_pi = F.log_softmax(pi_logits, dim=-1)
        return log_pi, mu, log_sigma

    def embed_node_feats(self, node_data):
        cont_feats = node_data[:, :self.feat_dim]
        node_labels = node_data[:, self.feat_dim].long()
        embed_cont = self.nodefeat_encoding(cont_feats)
        embed_label = self.nodelabel_encoding(node_labels)
        combined_embed = torch.cat([embed_cont, embed_label], dim=-1)
        return self.combiner(combined_embed)

    def predict_node_feats(self, state, node_data=None, label_mask=None):
        h, c = state
        N = h.shape[0]

        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std

        pred_logits = self.nodelabel_pred(h)

        if node_data is None:
            # --- Generation ---
            probs = F.softmax(pred_logits / self.label_temp, dim=-1)
            pred_labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
            label_embed = self.nodelabel_encoding(pred_labels)
            h_cond = torch.cat([h, label_embed], dim=-1)

            if self.cont_feat_dim > 0:
                log_pi, mu, log_sigma = self._mdn_params(h_cond)  # (N, F, K)
                pi = log_pi.exp()
                # Sample one component per (node, feature) independently.
                flat_pi = pi.reshape(N * self.cont_feat_dim, self.n_components)
                comp_idx = torch.multinomial(flat_pi, num_samples=1).squeeze(-1)
                comp_idx = comp_idx.view(N, self.cont_feat_dim)
                gather_idx = comp_idx.unsqueeze(-1)
                chosen_mu = mu.gather(-1, gather_idx).squeeze(-1)
                chosen_log_sigma = log_sigma.gather(-1, gather_idx).squeeze(-1)
                sampled_cont = chosen_mu + chosen_log_sigma.exp() * torch.randn_like(chosen_mu)
            else:
                sampled_cont = torch.empty(N, 0, device=h.device)

            if self.bin_feat_dim > 0:
                bin_logits = self.binfeat_pred(h_cond)
                sampled_bin = torch.bernoulli(torch.sigmoid(bin_logits))
            else:
                sampled_bin = torch.empty(N, 0, device=h.device)

            all_feats = self._assemble_features(sampled_cont, sampled_bin)
            return_data = torch.cat([all_feats, pred_labels.unsqueeze(-1).float()], dim=-1)
            state_update = self.embed_node_feats(return_data)
            ll = 0

        else:
            # --- Training ---
            target_all = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()
            label_embed = self.nodelabel_encoding(target_labels)
            h_cond = torch.cat([h, label_embed], dim=-1)

            target_cont = target_all[:, self.cont_idx] if self.cont_feat_dim > 0 else None
            target_bin = target_all[:, self.binary_idx] if self.bin_feat_dim > 0 else None

            # 1. MDN log-likelihood per (node, feature), averaged over features.
            if target_cont is not None and self.cont_feat_dim > 0:
                log_pi, mu, log_sigma = self._mdn_params(h_cond)      # (N, F, K)
                x = target_cont.unsqueeze(-1)                          # (N, F, 1)
                # Gaussian log-prob per component
                log_norm = -0.5 * math.log(2 * math.pi)
                log_comp = (log_norm - log_sigma
                            - 0.5 * ((x - mu) / log_sigma.exp()) ** 2)  # (N, F, K)
                log_mix = torch.logsumexp(log_pi + log_comp, dim=-1)   # (N, F)
                ll_cont = log_mix.sum() / self.cont_feat_dim
            else:
                ll_cont = torch.tensor(0.0, device=h.device)

            # 2. Binary BCE
            if target_bin is not None and self.bin_feat_dim > 0:
                bin_logits = self.binfeat_pred(h_cond)
                ll_bin = -F.binary_cross_entropy_with_logits(bin_logits, target_bin, reduction='sum')
                ll_bin = ll_bin / self.bin_feat_dim
            else:
                ll_bin = torch.tensor(0.0, device=h.device)

            # 3. Label cross-entropy
            if label_mask is not None and not label_mask.all():
                ll_label = -F.cross_entropy(pred_logits[label_mask], target_labels[label_mask], reduction='sum')
            else:
                ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')

            ll = self.w_cont * ll_cont + self.w_bin * ll_bin + self.w_label * ll_label

            self._ll_cont += ll_cont.item()
            self._ll_bin += ll_bin.item()
            self._ll_label += ll_label.item()

            # Scheduled sampling: feed the model's own predictions into the state
            # update on a fraction of steps. Loss above stays teacher-forced; only
            # the RNN state sees self-generated prefixes.
            if self.ss_prob > 0 and torch.rand(1).item() < self.ss_prob:
                with torch.no_grad():
                    ss_labels = torch.multinomial(
                        F.softmax(pred_logits, dim=-1), num_samples=1
                    ).squeeze(-1)
                    ss_label_embed = self.nodelabel_encoding(ss_labels)
                    ss_h_cond = torch.cat([h, ss_label_embed], dim=-1)
                    if self.cont_feat_dim > 0:
                        ss_cont = self._sample_mdn(ss_h_cond)
                    else:
                        ss_cont = torch.empty(h.shape[0], 0, device=h.device)
                    if self.bin_feat_dim > 0:
                        ss_bin = torch.bernoulli(torch.sigmoid(self.binfeat_pred(ss_h_cond)))
                    else:
                        ss_bin = torch.empty(h.shape[0], 0, device=h.device)
                ss_feats = self._assemble_features(ss_cont.detach(), ss_bin.detach())
                pred_data = torch.cat([ss_feats, ss_labels.unsqueeze(-1).float()], dim=-1)
                state_update = self.embed_node_feats(pred_data)
            else:
                state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, (h, c))
        return new_state, ll, return_data