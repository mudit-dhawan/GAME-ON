## Importing libraries
import os
import numpy as np
import pandas as pd

import dgl
from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn

import torch
from torch import nn
import torch.nn.functional as F

import networkx as nx
import json
import pickle

class GraphDataset(DGLDataset):
    def __init__(self, df, root_dir, image_id, text_id, 
                 image_vec_dir, text_vec_dir, dataset_name="GraphDataset"):
        """ Create Graph Dataset for Fake News Detection Task

        Args:
            df (pd.DataFrame)
            root_dir (str)
            image_id (str)
            text_id (str)
            image_vec_dir (str)
            text_vec_dir (str)
            dataset_name (str, optional). Defaults to "GraphDataset".
        """

        super(GraphDataset, self).__init__(name=dataset_name,
                                           verbose=True)

        ## Main CSV file
        self.df = df

        ## Base data folder
        self.root_dir = root_dir

        ## Unique text id list
        self.text_id = text_id
        
        ## directory that contains node embeddings for text graph
        self.text_vec_dir = text_vec_dir
        
    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass
    
    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """

        Args:
            idx : index of sample to be created

        Returns:
            dgl.graph: multimodal graph for a news post
            torch.tensor(): classification label corresponding to the news post
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## filename for the index
        file_name_text = self.df[self.text_id][idx]
    
        #### Adding text modality in Graph Dict

        ## Load node embeddings for tokens present in the text
        text_vec = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name_text}.npy')

        ## Load full image node embedding
        text_vec_full = np.load(f'{self.root_dir}{self.text_vec_dir}{file_name_text}_full_text.npy')
 
        all_text_vec = np.concatenate([text_vec_full, text_vec], axis=0)
        
        ## Creating a fully connected adjacency matrix
        text_adjMat = np.ones((all_text_vec.shape[0], all_text_vec.shape[0])).astype(float)
        text_pos_arr = np.where(text_adjMat > 0.5)
        
        ## Create tensor tuple for the dgl object 
        edges = (torch.tensor(np.concatenate((text_pos_arr[0], text_pos_arr[1])), dtype=torch.int64), 
               torch.tensor(np.concatenate((text_pos_arr[1], text_pos_arr[0])), dtype=torch.int64))

        g = dgl.graph(edges)

        ## find the label 
        if self.df['label'][idx] == 'real':
            label = 0
        elif self.df['label'][idx] == 'fake':
            label = 1
        
        ## Add node embedding to the multimodal graph
        g.ndata['features'] = torch.tensor(all_text_vec).float()
        
        return g, torch.tensor(label, dtype=torch.long)