import torch
from torch import nn
import timm
import torchvision


class Efficientnet(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0'):
        super(Efficientnet, self).__init__()
        # Efficientnet backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)

    def forward(self, x):
        x = self.backbone(x)
        return x