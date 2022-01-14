## Importing libraries
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import config

def model_metric(tn, fp, fn, tp):
    """Calculate Accuracy, precision, recall and F1-score    
    
    Args:   
        tn (float)  
        fn (float)
        fp (float)
        tp (float)
    """

    acc = ((tp+tn)/ (tp+fp+tn+fn))*100
    if tp==0 : 
        prec = 0
        rec = 0
        f1_score = 0
    else:
        ## calculate the Precision
        prec = (tp/ (tp+fp))*100

        ## calculate the Recall
        rec = (tp/ (tp + fn))*100
        
        ## calculate the F1-score
        f1_score = 2*prec*rec/(prec+rec)    

    return acc, prec, rec, f1_score
        

def train_func_epoch(epoch, model, data_loader, device, optimizer, scheduler):
    """Function for a single training epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (GraphDataLoader)
        device (str)
        optimizer (torch.optim)
        scheduler (torch.optim.lr_scheduler)
    """

    # Put the model into the training mode
    model.train()

    total_loss = 0
    
    ## To store values for each batch in an epoch 
    targets = []
    predictions = []
    
    ## Start a tqdm bar
    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:

        for step, batch in enumerate(single_epoch):

            single_epoch.set_description(f"Training- Epoch {epoch}")

            batched_graph, batch_labels = batch

            ## Load the inputs to the device
            batched_graph = batched_graph.to(device)
            batch_labels = batch_labels.to(device)

            # Perform a forward pass. This will return Multimodal vec and total loss.
            batch_logits = model(batched_graph)
            
            pred_multimodal = torch.argmax(batch_logits, dim=1).flatten().cpu().numpy()
            
            predictions.append(pred_multimodal)
            targets.append(batch_labels.cpu().numpy())
            
            ## Calculate the final loss
            loss = F.cross_entropy(batch_logits, batch_labels)
            
            total_loss += loss.item()
            
            loss.backward()
            
            if step % config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                
                ## Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                ## torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Zero out any previously calculated gradients
                model.zero_grad()

            ## Update tqdm bar
            single_epoch.set_postfix(train_loss=total_loss/(step+1))
    
    ## Create single vector for predictions and ground truth
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    ## Calculate performance metrics
    report = classification_report(targets, predictions, output_dict=True, labels=[0,1])
    
    ## Average out the loss 
    epoch_train_loss = total_loss / len(data_loader)
    
    return epoch_train_loss, report


def eval_func(model, data_loader, device, epoch=1):
    """Function for a single validation epoch

    Args:
        epoch (int): current epoch number
        model (nn.Module)
        data_loader (GraphDataLoader)
        device (str)
    """

    # Put the model into the training mode
    model.eval()

    total_loss = 0
    
    ## To store values for each batch in an epoch 
    targets = []
    predictions = []
    
    ## Start a tqdm bar
    with tqdm(data_loader, unit="batch", total=len(data_loader)) as single_epoch:

        for step, batch in enumerate(single_epoch):

            single_epoch.set_description(f"Evaluating- Epoch {epoch}")

            batched_graph, batch_labels = batch

            ## Load the inputs to the device
            batched_graph = batched_graph.to(device)
            batch_labels = batch_labels.to(device)
            
            with torch.no_grad():
                batch_logits = model(batched_graph)
            
            loss = F.cross_entropy(batch_logits, batch_labels)

            total_loss += loss.item()

            ## Update the tqdm bar
            single_epoch.set_postfix(loss=loss.item())

            # Finding predictions 
            pred_multimodal = torch.argmax(batch_logits, dim=1).flatten().cpu().numpy()
            
            predictions.append(pred_multimodal)
            targets.append(batch_labels.cpu().numpy())
    
    ## Create single vector for predictions and ground truth
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    
    ## Avg out the loss
    epoch_validation_loss = total_loss/len(data_loader)

    ## Find the performance metrics
    report = classification_report(targets, predictions, output_dict=True, labels=[0,1])

    ## Find the confusion metrics
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    
    ## Calculate Micro - metrics from own function
    acc, prec, rec, f1_score = model_metric(tn, fp, fn, tp)

    return epoch_validation_loss, report, acc, prec, rec, f1_score