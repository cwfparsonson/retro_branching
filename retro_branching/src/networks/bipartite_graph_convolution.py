import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np
import ml_collections
import copy
import json
import time

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need 
    to provide the exact form of the messages being passed.
    """
    def __init__(self,
                 aggregator='add',
                 emb_size=64,
                 include_edge_features=False):
        super().__init__(aggregator)

        self.include_edge_features = include_edge_features
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        if self.include_edge_features:
            self.feature_module_edge = torch.nn.Sequential(
                torch.nn.Linear(1, emb_size, bias=False)
            )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )
        
        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            # torch.nn.LayerNorm(emb_size, emb_size), # added
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
    # def forward(self, left_features, edge_indices, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        # output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                # node_features=(left_features, right_features), edge_features=edge_features)
        # output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                # node_features=(self.feature_module_left(left_features), self.feature_module_right(right_features)))
        if self.include_edge_features:
            edge_feats = self.feature_module_edge(edge_features)
        else:
            edge_feats = None
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                node_features=(self.feature_module_left(left_features), self.feature_module_right(right_features)), edge_features=edge_feats)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features=None):
    # def message(self, node_features_i, node_features_j):
        # output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           # # + self.feature_module_edge(edge_features) 
                                           # + self.feature_module_right(node_features_j))
        # output = self.feature_module_final(node_features_i + node_features_j)
        if edge_features is not None:
            output = self.feature_module_final(node_features_i + node_features_j + edge_features)
        else:
            output = self.feature_module_final(node_features_i + node_features_j)
        return output
