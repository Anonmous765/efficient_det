import torch
import torch.nn as nn
from torch import functional as F
from EfficientDet_Backbone import EfficientDetConfig
from Depthwise_convolution import DepthwiseSeparableConv


class ClassificationHead(nn.Module):
    def __init__(self, config, num_classes, num_anchors=9):
        super().__init__()
        out_channels = config.out_channels
        num_head_layers = config.num_head_layers
        NUM_LEVELS = 5

        # Shared conv weights — one module per layer, used on all 5 levels
        # We store depthwise and pointwise separately so we can insert
        # per-level BN between conv and activation
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
            for _ in range(num_head_layers)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
            for _ in range(num_head_layers)
        ])

        # Per-level BN: shape [num_levels][num_layers]
        # Each BN has its own running_mean, running_var, gamma, beta
        self.bn = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
                for _ in range(num_head_layers)
            ])
            for _ in range(NUM_LEVELS)
        ])

        self.act = nn.SiLU()

        # Final projection (shared across levels, no BN here)
        self.final_conv = nn.Conv2d(out_channels, num_classes * num_anchors, 1)

        # Bias initialization — makes model predict ~1% probability initially
        import math
        prior = 0.01
        nn.init.constant_(self.final_conv.bias, -math.log((1 - prior) / prior))

    def forward(self, features):
        outputs = []
        for level_idx, x in enumerate(features):  # P3, P4, P5, P6, P7
            for layer_idx in range(len(self.depthwise_convs)):
                x = self.depthwise_convs[layer_idx](x)   # shared weights
                x = self.pointwise_convs[layer_idx](x)   # shared weights
                x = self.bn[level_idx][layer_idx](x)     # per-level BN ✓
                x = self.act(x)
            outputs.append(self.final_conv(x))
        return outputs


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