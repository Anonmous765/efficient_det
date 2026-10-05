"""
Classification and box-regression heads, shared across all five pyramid levels.

Both heads apply the same stack of depthwise-separable conv layers to every
level (weights shared), with a separate BatchNorm per level, then a final 1x1
conv that predicts one output vector per anchor at each location.
"""
import math
import torch
import torch.nn as nn

from efficientdet.config import EfficientDetConfig


class ClassificationHead(nn.Module):
    """Predicts per-anchor class logits at every location of every level.

    The final conv's bias is set so that every anchor starts with a predicted
    probability of 0.01 (the RetinaNet prior). Otherwise the huge number of
    background anchors swamps the focal loss early in training.

    Args:
        config      : supplies out_channels and num_head_layers
        num_classes : number of object classes (no explicit background class)
        num_anchors : anchors per location (3 scales x 3 aspect ratios)
    """

    def __init__(self, config: EfficientDetConfig, num_classes: int, num_anchors: int = 9) -> None:
        super().__init__()
        out_channels = config.out_channels
        num_head_layers = config.num_head_layers

        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
            for _ in range(num_head_layers)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
            for _ in range(num_head_layers)
        ])

        # Per-level batch normalization: 5 levels × D layers
        self.bn = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
                for _ in range(num_head_layers)
            ])
            for _ in range(5)
        ])
        self.act = nn.SiLU()

        self.final_conv = nn.Conv2d(out_channels, num_classes * num_anchors, 1)
        prior = 0.01
        nn.init.constant_(self.final_conv.bias, -math.log((1 - prior) / prior))

    def forward(self, features):
        """features: list of 5 (B, C, H_i, W_i) -> list of 5 (B, num_anchors*num_classes, H_i, W_i) logits."""
        outputs = []
        for level_idx, x in enumerate(features):
            for layer_idx in range(len(self.depthwise_convs)):
                x = self.depthwise_convs[layer_idx](x)
                x = self.pointwise_convs[layer_idx](x)
                x = self.bn[level_idx][layer_idx](x)
                x = self.act(x)
            outputs.append(self.final_conv(x))
        return outputs


class BoxHead(nn.Module):
    """Predicts per-anchor box deltas (tx, ty, tw, th) at every location of every level.

    Same architecture as ClassificationHead, with 4 outputs per anchor. The
    deltas are relative to the anchor; see efficientdet.utils.box_ops.decode_boxes.

    Args:
        config      : supplies out_channels and num_head_layers
        num_anchors : anchors per location
    """

    def __init__(self, config: EfficientDetConfig, num_anchors: int = 9):
        super().__init__()
        out_channels = config.out_channels
        num_head_layers = config.num_head_layers

        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
            for _ in range(num_head_layers)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
            for _ in range(num_head_layers)
        ])

        # Per-level BN — 5 levels × D layers
        self.bn = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
                for _ in range(num_head_layers)
            ])
            for _ in range(5)
        ])
        self.act = nn.SiLU()
        self.final_conv = nn.Conv2d(out_channels, 4 * num_anchors, kernel_size=1)

    def forward(self, features: list[torch.Tensor]):
        """features: list of 5 (B, C, H_i, W_i) -> list of 5 (B, num_anchors*4, H_i, W_i) deltas."""
        out_features = []
        for level_idx, x in enumerate(features):
            for layer_idx in range(len(self.depthwise_convs)):
                x = self.depthwise_convs[layer_idx](x)
                x = self.pointwise_convs[layer_idx](x)
                x = self.bn[level_idx][layer_idx](x)
                x = self.act(x)
            out_features.append(self.final_conv(x))
        return out_features
