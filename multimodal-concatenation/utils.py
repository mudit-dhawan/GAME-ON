import random 
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import config, dataset

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    

        
    df_train = pd.read_csv(f'{config.root_dir}{config.me_train_csv_name}', sep='\t')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.me_test_csv_name}', sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "clean_image_id", "post_id",
                                config.me_image_vec_dir, config.me_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "clean_image_id", "post_id",
                                config.me_image_vec_dir, config.me_text_vec_dir)
    
    return dataset_train, dataset_test

def set_up_mediaeval2015():
    
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
    
    df_train = pd.read_csv(f'{config.root_dir}{config.we_train_csv_name}', sep='\t')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.we_test_csv_name}', sep='\t')
    df_test = df_test.dropna().reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "clean_image_id", "tweetId",
                                config.we_image_vec_dir, config.we_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "clean_image_id", "tweetId",
                                config.we_image_vec_dir, config.we_text_vec_dir)
    
    return dataset_train, dataset_test

def set_up_politifact():
    
    df_train = pd.read_csv(f'{config.root_dir}{config.pf_train_csv_name}')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.pf_test_csv_name}')
    df_test = df_test.dropna().reset_index(drop=True)
    
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    
    labels = df['label'].to_numpy()
    
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=7, stratify=labels)
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "image", "unique_id",
                                config.pf_image_vec_dir, config.pf_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "image", "unique_id", 
                                config.pf_image_vec_dir, config.pf_text_vec_dir)
    
    return dataset_train, dataset_test

def set_up_gossipcop():
    
    df_train = pd.read_csv(f'{config.root_dir}{config.gc_train_csv_name}')
    df_train = df_train.dropna().reset_index(drop=True)
    
    df_test = pd.read_csv(f'{config.root_dir}{config.gc_test_csv_name}')
    df_test = df_test.dropna().reset_index(drop=True)
    
    df = pd.concat([df_train, df_test])
    
    labels = df['label'].to_numpy()
    
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=7, stratify=labels)
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    dataset_train = dataset.GraphDataset(df_train, config.root_dir, "image", "unique_id", 
                                config.gc_image_vec_dir, config.gc_text_vec_dir)
    
    dataset_test = dataset.GraphDataset(df_test, config.root_dir, "image", "unique_id", 
                                config.gc_image_vec_dir, config.gc_text_vec_dir)
    
    return dataset_train, dataset_test

# def set_up_mediaeval():
    
#     df_train = pd.read_csv(f'{config.root_dir}{config.me_train_csv_name}', sep='\t')
#     df_train = df_train.dropna().reset_index(drop=True)
    
#     df_test = pd.read_csv(f'{config.root_dir}{config.me_test_csv_name}', sep='\t')
#     df_test = df_test.dropna().reset_index(drop=True)
    
#     dataset_train = dataset.GraphDataset(df_train, config.root_dir, "clean_image_id", "post_id",
#                                 config.me_image_simMat_dir, config.me_image_sentVec_dir, 
#                                 config.me_text_simMat_dir, config.me_text_sentVec_dir, 
#                                 topk=config.topk)
    
#     dataset_test = dataset.GraphDataset(df_test, config.root_dir, "clean_image_id", "post_id", 
#                                 config.me_image_simMat_dir, config.me_image_sentVec_dir, 
#                                 config.me_text_simMat_dir, config.me_text_sentVec_dir, 
#                                 topk=config.topk)
    
#     return dataset_train, dataset_test

# def set_up_politifact():
    
#     df_train = pd.read_csv(f'{config.root_dir}{config.pf_train_csv_name}')
#     df_train = df_train.dropna().reset_index(drop=True)
    
#     df_test = pd.read_csv(f'{config.root_dir}{config.pf_test_csv_name}')
#     df_test = df_test.dropna().reset_index(drop=True)
    
#     df = pd.concat([df_train, df_test]).reset_index(drop=True)
    
#     labels = df['label'].to_numpy()
    
#     df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=7, stratify=labels)
    
#     df_train = df_train.reset_index(drop=True)
#     df_test = df_test.reset_index(drop=True)
    
#     dataset_train = dataset.GraphDataset(df_train, config.root_dir, "image", "unique_id",
#                                 config.pf_image_simMat_dir, config.pf_image_sentVec_dir, 
#                                 config.pf_text_simMat_dir, config.pf_text_sentVec_dir, 
#                                 topk=config.topk)
    
#     dataset_test = dataset.GraphDataset(df_test, config.root_dir, "image", "unique_id", 
#                                 config.pf_image_simMat_dir, config.pf_image_sentVec_dir, 
#                                 config.pf_text_simMat_dir, config.pf_text_sentVec_dir, 
#                                 topk=config.topk)
    
#     return dataset_train, dataset_test

# def set_up_gossipcop():
    
#     df_train = pd.read_csv(f'{config.root_dir}{config.gc_train_csv_name}')
#     df_train = df_train.dropna().reset_index(drop=True)
    
#     df_test = pd.read_csv(f'{config.root_dir}{config.gc_test_csv_name}')
#     df_test = df_test.dropna().reset_index(drop=True)
    
#     df = pd.concat([df_train, df_test])
    
#     labels = df['label'].to_numpy()
    
#     df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=7, stratify=labels)
    
#     df_train = df_train.reset_index(drop=True)
#     df_test = df_test.reset_index(drop=True)
    
#     dataset_train = dataset.GraphDataset(df_train, config.root_dir, "image", "unique_id", 
#                                 config.gc_image_simMat_dir, config.gc_image_sentVec_dir, 
#                                 config.gc_text_simMat_dir, config.gc_text_sentVec_dir, 
#                                 topk=config.topk)
    
#     dataset_test = dataset.GraphDataset(df_test, config.root_dir, "image", "unique_id", 
#                                 config.gc_image_simMat_dir, config.gc_image_sentVec_dir, 
#                                 config.gc_text_simMat_dir, config.gc_text_sentVec_dir, 
#                                 topk=config.topk)
    
#     return dataset_train, dataset_test