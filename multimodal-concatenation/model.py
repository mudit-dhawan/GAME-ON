## Importing libraries
import os
import numpy as np
import pandas as pd

import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import conv

import torch
from torch import nn
import torch.nn.functional as F


class HeteroEmbeddingLayer(nn.Module):
    def __init__(self, 
                 in_feats, 
                 out_feats,
                 dropout_p):
        
        super(HeteroEmbeddingLayer, self).__init__()
        
        ## Initialize the the HeteroGraphConv 
        self.heterographlayers = nn.ModuleDict({ f"layer{i}": dglnn.HeteroGraphConv({
                                                    'image_edge'        : conv.GATConv(in_feats[i], out_feats[i], 
                                      num_heads=1, 
                                      feat_drop=0.4, 
                                      attn_drop=0.0, 
                                      residual=True, activation=nn.ELU(), bias=True),
                                                    'text_edge'         : conv.GATConv(in_feats[i], out_feats[i], 
                                      num_heads=1, 
                                      feat_drop=0.4, 
                                      attn_drop=0.0, 
                                      residual=True, activation=nn.ELU(), bias=True)
                                                },  aggregate='mean') for i in range(len(in_feats))})

        self.dropout_p = dropout_p
        self.num_hetero_layer = len(in_feats)

    def forward(self, g):
        
        x_features = {
            'image_node': g.nodes['image_node'].data['features'],
            'text_node': g.nodes['text_node'].data['features']
        }
        
        x_features['image_node'] = self.mm_embedding_space(x_features['image_node'])
        x_features['text_node'] = self.mm_embedding_space(x_features['text_node'])

        ## Loop over the number of gat layers
        for idx_layer, heterographlayer in self.heterographlayers.items():
            x_features = heterographlayer(g, x_features)
            
            for key, val in x_features.items():
                x_features[key] = torch.mean(val, dim=1)
        
        return x_features


## Hetero-Graph Classifier ##
class HeteroGraphClassifier(nn.Module):
    def __init__(self,
                 in_feats_embedding=[768, 512],
                 out_feats_embedding=[512, 256],
                 classifier_dims=[256], 
                 dropout_p=0.6,
                 n_classes=2):
        
        super(HeteroGraphClassifier, self).__init__()
        
        ## Inidividual GAT layers
        self.heteroembeddinglayers = HeteroEmbeddingLayer(in_feats=in_feats_embedding, 
                                                        out_feats=out_feats_embedding,
                                                         dropout_p=dropout_p)
        
        ## Fake News Detector
        self.classifier = nn.Sequential(
                            nn.Linear(512, 128),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(128, 2)
        )

        
    def forward(self, g):

        x_features = self.heteroembeddinglayers(g)

        with g.local_scope():
            g.nodes['image_node'].data['features'] = x_features['image_node']
            g.nodes['text_node'].data['features'] = x_features['text_node']
            
            # Calculate graph representation by average readout.
            hg = []
            for ntype in g.ntypes:
                hg.append(dgl.mean_nodes(g, 'features', ntype=ntype))
            
            ## Concatenate the mean node from each unimodal graph
            hg = torch.cat(hg, dim=1)

            return self.classifier(hg)