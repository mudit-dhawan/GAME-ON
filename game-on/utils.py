## Importing libraries
import random 
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import config, dataset

def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def set_up_mediaeval2015():
    """ 
    Loads the mediaeval graphical dataset.

    Download raw mediaeval dataset from: https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2015 

    Returns:
        DGLDataset: Train graph dataset
        DGLDataset: Test graph dataset
    """
    
    df_train = pd.read_csv(f'{config.root_dir}{config.me15_train_csv_name}', sep='\t')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.me15_test_csv_name}', sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "clean_image_id", "tweetId",
                                config.me15_image_vec_dir, config.me15_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "clean_image_id", "tweetId",
                                config.me15_image_vec_dir, config.me15_text_vec_dir)
    
    return dataset_train, dataset_test


def set_up_weibo():

    """ 
    Loads the weibo graphical dataset.

    Download raw mediaeval dataset from: https://github.com/yaqingwang/EANN-KDD18 

    Returns:
        DGLDataset: Train graph dataset
        DGLDataset: Test graph dataset
    """
    
    df_train = pd.read_csv(f'{config.root_dir}{config.we_train_csv_name}', sep='\t')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.we_test_csv_name}', sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "clean_image_id", "tweetId",
                                config.we_image_vec_dir, config.we_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "clean_image_id", "tweetId",
                                config.we_image_vec_dir, config.we_text_vec_dir)
    
    return dataset_train, dataset_test