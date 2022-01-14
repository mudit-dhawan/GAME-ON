## Importing libraries
import numpy as np
import pandas as pd
import math 

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import conv

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

import config, model, dataset, engine, utils

import wandb

wandb.init(project="", entity="", name="")

if __name__ == '__main__':
    
    ## Setup the dataset 

    dataset_name = "me15" ## me15, we
    utils.set_seed(5) 

    if dataset_name == "me15":  
        dataset_train, dataset_test = utils.set_up_mediaeval2015()
    elif dataset_name == "we":  
        dataset_train, dataset_test = utils.set_up_weibo()
    else:
        print("No Data")
    
    ## Setup the dataloaders
    dataloader_train = GraphDataLoader(
        dataset_train,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True)
    
    dataloader_test = GraphDataLoader(
        dataset_test,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Initialize the model
    gnn_model = model.HomoGraphClassifier(
                 in_feats_embedding=[768, 512],
                 out_feats_embedding=[512, 256],
                 classifier_dims=[128], 
                 dropout_p=0.4,
                 n_classes=2)


    gnn_model.to(device)
    
    ## Calculate number of train steps
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * config.epochs
    
    optimizer = AdamW(gnn_model.parameters(), lr=config.lr)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=num_train_steps
    )
    
    best_loss = np.inf
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch+1} ---------------------------------- \n")
        ## Training Loop
        train_loss, train_report = engine.train_func_epoch(epoch+1, gnn_model, dataloader_train, device, optimizer, scheduler)

        ## Validation loop
        val_loss, report, acc, prec, rec, f1_score = engine.eval_func(gnn_model, dataloader_test, device, epoch+1)
        
        print(f"\nEpoch: {epoch+1} | Training loss: {train_loss} | Validation Loss: {val_loss}")
        print()
        print("Train Report:")
        print(train_report)
        print()
        print("Validation Report:")
        print(report)
        print()
        print(f"Accuracy: {acc} | Micro Precision: {prec} | Micro Recall: {rec}, Micro F1-score: {f1_score} ")
        
        wandb.log({"train_loss": train_loss, "train-acc": train_report["accuracy"],"val-loss": val_loss, "val-prec": prec, "val-rec": rec, "val-f1score": f1_score, "val-acc": report["accuracy"]})
        
        print(f"\n----------------------------------------------------------------------------")