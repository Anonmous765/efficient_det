import math
import torch
import torch.nn as nn
from EfficientDet_Backbone import EfficientDetConfig


class ClassificationHead(nn.Module):
    def __init__(self, config: EfficientDetConfig, num_classes: int, num_anchors: int = 9) -> None:
        super().__init__()
        out_channels = config.out_channels
        num_head_layers = config.num_head_layers

        # Shared depthwise and pointwise convolutions across all feature levels
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
            for _ in range(num_head_layers)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
            for _ in range(num_head_layers)
        ])

        # Per-level batch normalization
        self.bn = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
                for _ in range(num_head_layers)
            ])
            for _ in range(5)
        ])
        self.act = nn.SiLU()

        # Final projection (shared across levels, no BN here)
        self.final_conv = nn.Conv2d(out_channels, num_classes * num_anchors, 1)

        # Bias initialization — makes model predict ~1% probability initially
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
        out_channels = config.out_channels
        num_head_layer = config.num_head_layers

        # Shared conv weights
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
            for _ in range(num_head_layer)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
            for _ in range(num_head_layer)
        ])

        # Per-level BN — 5 levels × D layers
        self.bn = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
                for _ in range(num_head_layer)
            ])
            for _ in range(5)
        ])

        self.act = nn.SiLU()
        self.final_conv = nn.Conv2d(out_channels, 4 * num_anchors, kernel_size=1)

    def forward(self, features: list[torch.Tensor]):
        out_features = []
        for level_idx, x in enumerate(features):
            for layer_idx in range(len(self.depthwise_convs)):
                x = self.depthwise_convs[layer_idx](x)
                x = self.pointwise_convs[layer_idx](x)
                x = self.bn[level_idx][layer_idx](x)  # per-level BN
                x = self.act(x)
            out_features.append(self.final_conv(x))
        return out_features