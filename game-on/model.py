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

    
class MModel(nn.Module):
    def __init__(self):
            
        super(MModel, self).__init__()
        
        ## Projection layer
        self.mm_embedding_space = nn.Sequential(
                            nn.Linear(768, 768),
                            nn.ELU(),
                            nn.Dropout(0.4)
        )
        
        ## GAT Layer
        self.homoembeddinglayers_1 = conv.GATConv(768, 256, num_heads=1, feat_drop=0.4, attn_drop=0.0, residual=True, activation=nn.ELU(), bias=True)

        ## Fake News Detector
        self.classifier = nn.Sequential(
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(0.4),
                            nn.Linear(128, 2)
        )
        
    def forward(self, g):
        x = g.ndata['features']
        
        ## Project to common multimodal space
        x= self.mm_embedding_space(x)
        
        ## Apply single gat layer
        x = torch.mean(self.homoembeddinglayers_1(g, x), dim=1)

        g.ndata['features'] = x
        
        ## Take mean representation of the multimodal graph
        x = dgl.mean_nodes(g, 'features')

        ## Classify
        x = self.classifier(x)

        return x