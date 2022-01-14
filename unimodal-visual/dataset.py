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

        ## Unique image id list
        self.image_id = image_id
        
        ## directory that contains node embeddings for image graph
        self.image_vec_dir = image_vec_dir

        ## to resize the imagefeature vector from pre-trained model
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(768)
        
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

        ## filenames for the index
        file_name_image = self.df[self.image_id][idx].split(".")[0]
    
        #### Adding image modality in Graph Dict

        ## Load full image node embedding
        image_vec_full = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name_image}_full_image.npy')
        
        ## Load node embeddings for objects present in the image
        try:
            image_vec = np.load(f'{self.root_dir}{self.image_vec_dir}{file_name_image}.npy')
            all_image_vec = np.concatenate([image_vec_full, image_vec], axis=0)
        except:
            all_image_vec = image_vec_full
        
        ## Creating a fully connected adjacency matrix
        image_adjMat = np.ones((all_image_vec.shape[0], all_image_vec.shape[0])).astype(float)
        image_pos_arr = np.where(image_adjMat > 0.5)
        
        ## Create tensor tuple for the dgl object 
        edges = (torch.tensor(np.concatenate((image_pos_arr[0], image_pos_arr[1])), dtype=torch.int64), 
                 torch.tensor(np.concatenate((image_pos_arr[1], image_pos_arr[0])), dtype=torch.int64))
        
        g = dgl.graph(edges)

        ## find the label 
        if self.df['label'][idx] == 'real':
            label = 0
        elif self.df['label'][idx] == 'fake':
            label = 1
        
        ## Add node embedding to the multimodal graph
        ## Resize the image vectors to match the text embedding dimension
        g.ndata['features'] = self.adaptive_pooling(torch.tensor(all_image_vec).float().unsqueeze(0)).squeeze(0)
        
        return g, torch.tensor(label, dtype=torch.long)