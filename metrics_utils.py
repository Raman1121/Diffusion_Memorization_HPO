import skimage, torch, torchvision
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn   
import torchxrayvision as xrv
import random
from torchmetrics.image.fid import FrechetInceptionDistance


def get_images_tensor_from_paths(image_paths):
    transform = torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
            torchvision.transforms.ToTensor(),
        ]
    )
    imgs = []
    for path in image_paths:
        img = skimage.io.imread(path).T  # transpose to get the right orientation
        img = xrv.datasets.normalize(img, 255)
        img = img[None, ...] if len(img.shape) == 2 else img
        img = transform(img)
        imgs.append(img)

    return torch.stack(imgs).squeeze().unsqueeze(1)

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, model_type='mimic_densenet'):
        
        super(FeatureExtractor, self).__init__() 
        self.model = model
        self.model_type = model_type
        
    def forward(self,x):
        if(self.model_type == 'mimic_densenet'):
            return self.model.features(x).mean([-1,-2]) 
        elif(self.model_type == 'medclip_vit' or self.model_type == 'medclip_resnet' or self.model_type == 'biomedclip'):
            return self.model(x)
        else:
            raise NotImplementedError("Encoder type not supported")

# def run_predictions(real_images, synthetic_images, device):

#     batch_size = 64
#     print("Running Predictions using MIMIC DenseNet")
#     model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch").to(device)

#     real_features = []
#     real_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(real_images), batch_size=batch_size
#     )
#     for batch in real_loader:
#         real_feature = model.features(batch[0].to(device))
#         real_feature = torch.mean(real_feature, (-2, -1))
#         real_features.append(real_feature.detach().cpu())

#     real_features = torch.cat(real_features)

#     synth_features = []
#     synth_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(synthetic_images), batch_size=batch_size
#     )
#     for batch in synth_loader:
#         synth_feature = model.features(batch[0].to(device))
#         synth_feature = torch.mean(synth_feature, (-2, -1))
#         synth_features.append(synth_feature.detach().cpu())

#     synth_features = torch.cat(synth_features)

#     return real_features, synth_features

def fid_metric(real_images, synthetic_images, features=64, device=None, encoder_type='mimic_densenet'):
    if(isinstance(features, int)):
        print("Using InceptionV3 feature Layer {}".format(features))
    elif(isinstance(features, nn.Module)):
        print("Using a custom feature extractor for FID calculation")
    else:
        print("Invalid features provided")
        return np.nan

    fid = FrechetInceptionDistance(feature=features).to(device)
    fid.update(real_images, real=True)
    fid.update(synthetic_images, real=False)
    
    return fid.compute().cpu().item()

def compute_fid(real_image_tensors, synth_image_tensors, device):
    real_image_tensors = real_image_tensors.to(torch.float)
    synth_image_tensors = synth_image_tensors.to(torch.float)
    real_image_tensors = real_image_tensors.to(device)
    synth_image_tensors = synth_image_tensors.to(device)

    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch").to(device)
    feature_extractor = FeatureExtractor(model, model_type='mimic_densenet')

    fid = fid_metric(real_image_tensors, synth_image_tensors, features=feature_extractor, device=device)

    return fid

