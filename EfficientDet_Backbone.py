import torch
import torch.nn as nn
import timm


class EfficientDetConfig:
    def __init__(self, phi: int = 0):
        if not (0 <= phi <= 7):
            raise ValueError(f"phi must be 0–7, got {phi}")

        self.phi = phi
        self.out_channels = int(round(64 * (1.35 ** phi) / 8)) * 8
        self.num_bifpn_layers = 3 + phi
        self.num_head_layers = 3 + phi // 3
        self.input_resolution = 512 + phi * 128
        self.backbone_name = [
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b6",
        ][phi]

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

        # 1x1 projections for C3, C4, C5
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, config.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(config.out_channels),
            )
            for in_ch in in_channels
        ])

        # P6 and P7 generators registered as proper submodules
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