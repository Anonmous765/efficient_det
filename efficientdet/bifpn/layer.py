import torch.nn as nn
import torch.nn.functional as F

from efficientdet.bifpn.fusion import FastNormalizedFusion
from efficientdet.utils.conv import DepthwiseSeparableConv
from efficientdet.config import EfficientDetConfig


class TopDownNodes(nn.Module):
    """Top-down pathway: fuses P7->P6->P5->P4 via weighted addition and upsampling."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.p6_td_fuse = FastNormalizedFusion(num_inputs=2)
        self.p5_td_fuse = FastNormalizedFusion(num_inputs=2)
        self.p4_td_fuse = FastNormalizedFusion(num_inputs=2)

        self.p6_td_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.p5_td_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.p4_td_conv = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, p3, p4, p5, p6, p7):
        p6_td = self.p6_td_conv(self.p6_td_fuse(p6, F.interpolate(p7, size=p6.shape[-2:], mode='nearest')))
        p5_td = self.p5_td_conv(self.p5_td_fuse(p5, F.interpolate(p6_td, size=p5.shape[-2:], mode='nearest')))
        p4_td = self.p4_td_conv(self.p4_td_fuse(p4, F.interpolate(p5_td, size=p4.shape[-2:], mode='nearest')))
        return p3, p4, p5, p6, p7, p4_td, p5_td, p6_td


class OutNodes(nn.Module):
    """Bottom-up output pathway: combines original and top-down features to produce P3-P7 outputs."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.p3_out_fuse = FastNormalizedFusion(num_inputs=2)
        self.p4_out_fuse = FastNormalizedFusion(num_inputs=3)
        self.p5_out_fuse = FastNormalizedFusion(num_inputs=3)
        self.p6_out_fuse = FastNormalizedFusion(num_inputs=3)
        self.p7_out_fuse = FastNormalizedFusion(num_inputs=2)

        self.p3_out_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.p4_out_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.p5_out_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.p6_out_conv = DepthwiseSeparableConv(out_channels, out_channels)
        self.p7_out_conv = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, p3, p4, p5, p6, p7, p4_td, p5_td, p6_td):
        p3_out = self.p3_out_conv(self.p3_out_fuse(p3, F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')))
        p4_out = self.p4_out_conv(self.p4_out_fuse(p4, p4_td, F.max_pool2d(p3_out, kernel_size=2, stride=2)))
        p5_out = self.p5_out_conv(self.p5_out_fuse(p5, p5_td, F.max_pool2d(p4_out, kernel_size=2, stride=2)))
        p6_out = self.p6_out_conv(self.p6_out_fuse(p6, p6_td, F.max_pool2d(p5_out, kernel_size=2, stride=2)))
        p7_out = self.p7_out_conv(self.p7_out_fuse(p7, F.max_pool2d(p6_out, kernel_size=2, stride=2)))
        return p3_out, p4_out, p5_out, p6_out, p7_out


class BiFPNLayer(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.top_down = TopDownNodes(out_channels)
        self.out_nodes = OutNodes(out_channels)

    def forward(self, p3, p4, p5, p6, p7):
        p3, p4, p5, p6, p7, p4_td, p5_td, p6_td = self.top_down(p3, p4, p5, p6, p7)
        return self.out_nodes(p3, p4, p5, p6, p7, p4_td, p5_td, p6_td)


class BiFPN(nn.Module):
    def __init__(self, config: EfficientDetConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiFPNLayer(config.out_channels) for _ in range(config.num_bifpn_layers)]
        )

    def forward(self, p3, p4, p5, p6, p7):
        for layer in self.layers:
            p3, p4, p5, p6, p7 = layer(p3, p4, p5, p6, p7)
        return p3, p4, p5, p6, p7
