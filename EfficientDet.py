import torch
import torch.nn as nn
from EfficientDet_Backbone import EfficientDetConfig, EfficientDetBackbone
from BiFPN_layer import BiFPN
from efficientdet_heads import ClassificationHead, BoxHead


class EfficientDet(nn.Module):
    def __init__(self, config: EfficientDetConfig, num_classes: int, num_anchors: int = 9):
        super().__init__()
        self.backbone = EfficientDetBackbone(config)
        self.bifpn = BiFPN(config)
        self.class_head = ClassificationHead(config, num_classes, num_anchors)
        self.box_head = BoxHead(config, num_anchors)

    def forward(self, x):
        p3, p4, p5, p6, p7 = self.backbone(x)
        p3, p4, p5, p6, p7 = self.bifpn(p3, p4, p5, p6, p7)
        features = [p3, p4, p5, p6, p7]
        class_outputs = self.class_head(features)
        box_outputs = self.box_head(features)
        return class_outputs, box_outputs


if __name__ == "__main__":
    config = EfficientDetConfig(phi=0)
    model = EfficientDet(config, num_classes=80)
    x = torch.randn(1, 3, 512, 512)
    class_outputs, box_outputs = model(x)
    print("Classification outputs:")
    for i, out in enumerate(class_outputs):
        print(f"  P{i+3}: {tuple(out.shape)}")
    print("Box outputs:")
    for i, out in enumerate(box_outputs):
        print(f"  P{i+3}: {tuple(out.shape)}")
