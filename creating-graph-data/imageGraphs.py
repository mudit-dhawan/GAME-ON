## importing libraries
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import pandas as pd
import os

def store_features(od_model, feature_model, images_list, root_dir, image_dir, store_dir):
    
    lengths = []
    
    for idx in tqdm(range(len(images_list))):
        
        image_name = images_list[idx]
        unique_id = image_name.split(".")[0]
        
        image_name = f"{root_dir}{image_dir}{image_name}"

        try:
            image = Image.open(image_name).convert('RGB')
        except:
            print("Image name:", image_name)
            continue

        image_t = transform(image).to(device)
        image_t = image_t.unsqueeze(0) # add a batch dimension
        image_t = image_t[:,:3, :, :]
        
        with torch.no_grad():
            outputs = od_model(image_t) # get the predictions on the image

        image_features = []

        transformed_image = image_transform(image).unsqueeze(0)
        transformed_image = transformed_image.to(device)
        transformed_image = transformed_image[:,:3, :, :]
        
        with torch.no_grad():
            feature = feature_model(transformed_image)
        
        ## Find features for the whole image
        feature = feature.detach().cpu().numpy()[:,:,0,0] 

        ## Save whole image representation
        filename = f'{root_dir}{store_dir}{unique_id}_full_image.npy'
        np.save(filename, feature)
        

        ## Loop over the bounding boxes detected
        for i in range(len(outputs[0]['boxes'])):
            box = outputs[0]['boxes'][i]

            ## create a cropped image
            x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cropped_image = image.convert("RGB").crop((x1, y1, x2, y2))
            
            ## Transform the cropped image
            transformed_image = image_transform(cropped_image).unsqueeze(0)
            transformed_image = transformed_image.to(device)
            transformed_image = transformed_image[:,:3, :, :]
            
            ## Find feature for each object
            with torch.no_grad():
                feature = feature_model(transformed_image)

            image_features.append(feature.detach().cpu().numpy()[:,:,0,0]) 
        
        lengths.append(len(image_features))

        ### Store object level representations
        try:
            image_features = np.concatenate(image_features, axis=0)
        except:
            print("Image name:", image_name)
            continue

        filename = f'{root_dir}{store_dir}{unique_id}.npy'
        np.save(filename, image_features)
        
    return lengths

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Load object detection model
    fatser_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    ## Set the transforms for the images
    image_transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    fatser_rcnn.eval().to(device)

    ## Load the model for feature representations
    resnet = torchvision.models.resnet50(pretrained=True) 
    feature_resnet = torch.nn.Sequential(*list(resnet.children())[:-1]) ## output size 2048
    feature_resnet.eval().to(device)

    ## Base directory for the data
    root_dir = ""

    ## Directory location for train and test images
    images_train = ""
    images_test = ""

    ## Create a list of names for the images
    images_list_train = os.listdir(f'{root_dir}{images_train}')
    images_list_test = os.listdir(f'{root_dir}{images_test}')

    ## Directory to store the node embeddings for each image
    store_dir = ""
    
    ## store graph data for training images
    lengths = store_features(fatser_rcnn, feature_resnet, images_list_train, root_dir, images_train, store_dir)
    
    ## store graph data for testing images
    lengths = store_features(fatser_rcnn, feature_resnet, images_list_test, root_dir, images_test, store_dir)