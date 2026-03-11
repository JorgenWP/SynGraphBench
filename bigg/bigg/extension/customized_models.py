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

    def __init__(self, args, feat_dim, num_classes):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # 1. Continuous feature encoders/decoders
        self.nodefeat_encoding = MLP(feat_dim, [2 * args.embed_dim, args.embed_dim])
        self.nodefeat_pred = MLP(args.embed_dim, [2 * args.embed_dim, feat_dim])
        
        # 2. Discrete label encoders/decoders
        self.nodelabel_encoding = nn.Embedding(num_classes, args.embed_dim)
        self.nodelabel_pred = MLP(args.embed_dim, [2 * args.embed_dim, num_classes])
        
        # 3. Combiner to fuse both embeddings back to the expected args.embed_dim
        self.combiner = nn.Linear(args.embed_dim * 2, args.embed_dim)
        
        self.node_state_update = nn.LSTMCell(args.embed_dim, args.embed_dim)

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
        h, _ = state
        
        # Predict continuous features and class logits
        pred_cont = self.nodefeat_pred(h)
        pred_logits = self.nodelabel_pred(h)
        
        if node_data is None:
            # Generation mode
            ll = 0
            
            # Greedily pick the highest probability class
            pred_labels = torch.argmax(pred_logits, dim=-1, keepdim=True).float()
            
            # Concatenate predictions back into a single tensor of shape [batch_size, feat_dim + 1]
            return_data = torch.cat([pred_cont, pred_labels], dim=-1)
            
            state_update = self.embed_node_feats(return_data)
        else:
            # Training mode
            target_cont = node_data[:, :self.feat_dim]
            target_labels = node_data[:, self.feat_dim].long()
            
            # 1. Likelihood for continuous features (Negative MSE)
            ll_cont = -(target_cont - pred_cont) ** 2
            ll_cont = torch.sum(ll_cont)
            
            # 2. Likelihood for discrete labels (Negative Cross-Entropy)
            ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')
            
            # Note: You may want to multiply one of the likelihoods by a balancing scalar (e.g. 10.0) 
            # if the scales of the two losses differ too drastically.
            ll = ll_cont + ll_label
            
            state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, state)
        
        return new_state, ll, return_data

class BiggWithConditionedFeats(RecurTreeGen):

    def __init__(self, args, feat_dim, num_classes):
        super().__init__(args)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
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

    def embed_node_feats(self, node_data):
        cont_feats = node_data[:, :self.feat_dim]
        node_labels = node_data[:, self.feat_dim].long()
        
        embed_cont = self.nodefeat_encoding(cont_feats)
        embed_label = self.nodelabel_encoding(node_labels)
        
        combined_embed = torch.cat([embed_cont, embed_label], dim=-1)
        return self.combiner(combined_embed)

    def predict_node_feats(self, state, node_data=None):
        h, _ = state
        
        # Step 1: Predict class logits from the hidden state 'h'
        pred_logits = self.nodelabel_pred(h)
        
        if node_data is None:
            # --- Generation Mode ---
            ll = 0
            
            # Get the predicted label
            pred_labels = torch.argmax(pred_logits, dim=-1)
            
            # Step 2: Embed the predicted label
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
            ll_cont = -(target_cont - pred_cont) ** 2
            ll_cont = torch.sum(ll_cont)
            
            ll_label = -F.cross_entropy(pred_logits, target_labels, reduction='sum')
            
            ll = ll_cont + ll_label
            
            state_update = self.embed_node_feats(node_data)
            return_data = node_data

        new_state = self.node_state_update(state_update, state)
        
        return new_state, ll, return_data