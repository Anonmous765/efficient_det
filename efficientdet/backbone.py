import torch
import torch.nn as nn
import timm

from efficientdet.config import EfficientDetConfig


class EfficientDetBackbone(nn.Module):
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
