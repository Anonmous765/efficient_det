"""
EfficientNet backbone that produces the five-level feature pyramid P3-P7.

Run directly (`python -m efficientdet.backbone`) to print the feature-map
shapes for a 512x512 input.
"""
import torch
import torch.nn as nn
import timm

from efficientdet.config import EfficientDetConfig


class EfficientDetBackbone(nn.Module):
    """ImageNet-pretrained EfficientNet (via timm) plus the extra P6/P7 levels.

    Takes the stride-8/16/32 stages (C3-C5), projects each to
    `config.out_channels` with a 1x1 conv + BN, and builds P6 and P7 by
    downsampling P5 twice with stride-2 3x3 convs.
    """
    FEATURE_INDICES = (2, 3, 4)

    def __init__(self, config: EfficientDetConfig):
        super().__init__()
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=self.FEATURE_INDICES,
        )
        self.out_channels = config.out_channels

        in_channels = self.backbone.feature_info.channels()

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, config.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.out_channels),
            )
            for in_ch in in_channels
        ])

        self.p6_gen = nn.Sequential(
            nn.Conv2d(config.out_channels, config.out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.out_channels),
            nn.SiLU(),
        )
        self.p7_gen = nn.Sequential(
            nn.Conv2d(config.out_channels, config.out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        """x: (B, 3, H, W) -> (p3, p4, p5, p6, p7), strides 8-128, each with out_channels."""
        c3, c4, c5 = self.backbone(x)
        p3 = self.proj[0](c3)
        p4 = self.proj[1](c4)
        p5 = self.proj[2](c5)
        p6 = self.p6_gen(p5)
        p7 = self.p7_gen(p6)
        return p3, p4, p5, p6, p7


if __name__ == '__main__':
    config = EfficientDetConfig(phi=0)
    model = EfficientDetBackbone(config=config)
    features = model(torch.randn(1, 3, 512, 512))
    for i, f in enumerate(features, start=3):
        print(f"P{i}: {f.shape}")
