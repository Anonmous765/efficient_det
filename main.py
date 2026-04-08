import torch
from torch import nn
import timm
import torchvision


class EfficientDetBackbone(nn.Module):
    FEATURE_INDICES = (2, 3, 4)

    def __init__(self, backbone_name: str = "efficientnet_b0", out_channels: int = 160):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=self.FEATURE_INDICES,
        )

        in_channels = self.backbone.feature_info.channels()

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            for in_ch in in_channels
        ])

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3 = self.proj[0](c3)
        p4 = self.proj[1](c4)
        p5 = self.proj[2](c5)
        features = (p3, p4, p5)
        return features

if __name__ == '__main__':
    model = EfficientDetBackbone()
    x, y ,z = model(torch.randn(1, 3, 512, 512))
    print(x.shape)
    print(y.shape)
    print(z.shape)