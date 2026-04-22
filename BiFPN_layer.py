import torch.nn as nn
import torch.nn.functional as F

from BiFPN_fusion import FastNormalizedFusion
from Depthwise_convolution import DepthwiseSeparableConv


class TopDownNodes(nn.Module):
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
