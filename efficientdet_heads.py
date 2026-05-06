import torch
import torch.nn as nn
from Depthwise_convolution import DepthwiseSeparableConv
from EfficientDet_Backbone import EfficientDetConfig


class ClassificationHead(nn.Module):
    def __init__(self, config: EfficientDetConfig, num_classes: int, num_anchors: int = 9):
        super().__init__()

        # Initialize convolutional layers using DepthwiseSeparableConv
        self.conv_layers = nn.ModuleList(
            [
                DepthwiseSeparableConv(config.out_channels, config.out_channels)
                for _ in range(config.num_head_layers)
            ]
        )

        # Final convolution layer to produce output features
        self.conv = nn.Conv2d(config.out_channels, num_classes * num_anchors, kernel_size=1)

    def forward(self, features: list[torch.Tensor]):
        out_features = []

        for feature in features:
            for layer in self.conv_layers:
                feature = layer(feature)
            out_features.append(self.conv(feature))

        return out_features

class BoxHead(nn.Module):
    def __init__(self, config: EfficientDetConfig, num_anchors: int = 9):
        super().__init__()

        # Initialize convolutional layers using DepthwiseSeparableConv
        self.conv_layers = nn.ModuleList(
            [
                DepthwiseSeparableConv(config.out_channels, config.out_channels)
                for _ in range(config.num_head_layers)
            ]
        )
        self.conv = nn.Conv2d(config.out_channels, 4 * num_anchors, kernel_size=1)

    def forward(self, features: list[torch.Tensor]):
        out_features = []

        for feature in features:
            for layer in self.conv_layers:
                feature = layer(feature)
            out_features.append(self.conv(feature))

        return out_features