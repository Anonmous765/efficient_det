import torch
import torch.nn as nn

from efficientdet.config import EfficientDetConfig
from efficientdet.backbone import EfficientDetBackbone
from efficientdet.bifpn import BiFPN
from efficientdet.heads import ClassificationHead, BoxHead
from efficientdet.utils.anchors import AnchorGenerator


class EfficientDet(nn.Module):
    def __init__(self, config: EfficientDetConfig, num_classes: int, num_anchors: int = 9):
        super().__init__()
        self.backbone = EfficientDetBackbone(config)
        self.bifpn = BiFPN(config)
        self.class_head = ClassificationHead(config, num_classes, num_anchors)
        self.box_head = BoxHead(config, num_anchors)
        self.anchor_gen = AnchorGenerator()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        p3, p4, p5, p6, p7 = self.backbone(x)
        p3, p4, p5, p6, p7 = self.bifpn(p3, p4, p5, p6, p7)
        features = [p3, p4, p5, p6, p7]

        class_outputs = self.class_head(features)
        box_outputs   = self.box_head(features)

        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in features]
        anchors = self.anchor_gen(feature_map_sizes, device=x.device)

        B = x.shape[0]
        class_preds = self._flatten_outputs(class_outputs, B, self.num_classes)
        box_preds   = self._flatten_outputs(box_outputs,   B, 4)

        return class_preds, box_preds, anchors

    @staticmethod
    def _flatten_outputs(maps, B, last_dim):
        """
        maps     : list of (B, num_anchors*last_dim, H, W)
        Returns  : (B, N_total, last_dim)
        """
        all_preds = []
        for feat in maps:
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.reshape(B, -1, last_dim)
            all_preds.append(feat)
        return torch.cat(all_preds, dim=1)


if __name__ == "__main__":
    config = EfficientDetConfig(phi=0)
    model = EfficientDet(config, num_classes=80)
    x = torch.randn(1, 3, 512, 512)
    class_preds, box_preds, anchors = model(x)
    print(f"class_preds : {tuple(class_preds.shape)}")
    print(f"box_preds   : {tuple(box_preds.shape)}")
    print(f"anchors     : {tuple(anchors.shape)}")
