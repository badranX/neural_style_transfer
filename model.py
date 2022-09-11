import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.selected_layers = [3, 8, 15, 22]
        self.vgg = models.vgg16(pretrained=True).features

    def forward(self, x):
        layers_feats = []
        for layer_num, layer in self.vgg._modules.items():
            x = layer(x)
            if int(layer_num) in self.selected_layers:
                layers_feats.append(x)
        return layers_feats

