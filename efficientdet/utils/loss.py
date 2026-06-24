import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class EfficientDetLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, cls_logits, box_pred, cls_targets, box_targets, positive_mask):
        # cls_logits: (B, num_anchors, num_classes)
        # box_pred:   (B, num_anchors, 4)
        # positive_mask: (B, num_anchors) bool

        loss_cls = sigmoid_focal_loss(
            cls_logits, cls_targets,
            alpha=self.alpha, gamma=self.gamma, reduction="sum"
        )
        loss_reg = F.smooth_l1_loss(
            box_pred[positive_mask],
            box_targets[positive_mask],
            beta=self.beta, reduction="sum"
        )
        num_pos = positive_mask.sum().clamp(min=1).float()
        return (loss_cls + loss_reg) / num_pos