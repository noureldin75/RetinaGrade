import torch
import torch.nn as nn
import torchvision.models as models
from custom_layers import GeM
def build_model(model_name="efficientnet_b3", num_classes=5, freeze_features=True):
    model_func = getattr(models, model_name)
    model = model_func(weights="DEFAULT")
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False




    model.avgpool = GeM()

    in_features=model.classifier[1].in_features
    model.classifier[1]=nn.Linear(in_features=in_features,out_features=num_classes)

    return model
