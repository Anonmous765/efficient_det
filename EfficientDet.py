import torch
import torch.nn as nn
from EfficientDet_Backbone import EfficientDetConfig, EfficientDetBackbone
from BiFPN_layer import BiFPN
from efficientdet_heads import ClassificationHead, BoxHead
from anchor_generator import AnchorGenerator


class EfficientDet(nn.Module):
    def __init__(self, config, num_classes, num_anchors=9):
        super().__init__()
        self.backbone = EfficientDetBackbone(config)
        self.bifpn = BiFPN(config)
        self.class_head = ClassificationHead(config, num_classes, num_anchors)
        self.box_head = BoxHead(config, num_anchors)
        self.anchor_gen = AnchorGenerator()   # ← add this
        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        p3, p4, p5, p6, p7 = self.backbone(x)
        p3, p4, p5, p6, p7 = self.bifpn(p3, p4, p5, p6, p7)
        features = [p3, p4, p5, p6, p7]

        class_outputs = self.class_head(features)  # list of 5: (B, A*C, H, W)
        box_outputs   = self.box_head(features)    # list of 5: (B, A*4, H, W)

        # ── Derive feature map sizes ──────────────────────────────────────
        # Each output has shape (B, channels, H, W) — grab H and W
        feature_map_sizes = [(f.shape[2], f.shape[3]) for f in features]
        # e.g. [(64,64), (32,32), (16,16), (8,8), (4,4)] for 512px input

        # ── Generate anchors on the correct device ────────────────────────
        device = x.device  # follows the input tensor's device automatically
        anchors = self.anchor_gen(feature_map_sizes, device=device)
        # anchors: (49104, 4)  — (cx, cy, w, h)

        # ── Flatten head outputs to match anchor ordering ─────────────────
        B = x.shape[0]
        class_preds = self._flatten_outputs(class_outputs, B, self.num_classes)
        box_preds   = self._flatten_outputs(box_outputs,   B, 4)
        # class_preds: (B, 49104, num_classes)
        # box_preds:   (B, 49104, 4)

        return class_preds, box_preds, anchors

    @staticmethod
    def _flatten_outputs(maps, B, last_dim):
        """
        maps     : list of tensors, each (B, num_anchors*last_dim, H, W)
        last_dim : num_classes or 4

        Returns  : (B, N_total, last_dim)

        The permute+reshape ensures the anchor variant index is innermost,
        matching the anchor generator's loop order.
        """
        all_preds = []
        for feat in maps:
            # feat:   (B, A*last_dim, H, W)
            # → permute: (B, H, W, A*last_dim)
            # → reshape: (B, H*W*A, last_dim)
            feat = feat.permute(0, 2, 3, 1)
            feat = feat.reshape(B, -1, last_dim)
            all_preds.append(feat)
        return torch.cat(all_preds, dim=1)  # (B, N_total, last_dim)

if __name__ == "__main__":
    config = EfficientDetConfig(phi=0)
    model = EfficientDet(config, num_classes=80)
    x = torch.randn(1, 3, 512, 512)
    class_preds, box_preds, anchors = model(x)
    print(f"class_preds : {tuple(class_preds.shape)}")
    print(f"box_preds   : {tuple(box_preds.shape)}")
    print(f"anchors     : {tuple(anchors.shape)}")
