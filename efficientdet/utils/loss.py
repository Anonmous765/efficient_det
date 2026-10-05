"""EfficientDet training loss: sigmoid focal loss for classification + smooth-L1 for boxes."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class EfficientDetLoss(nn.Module):
    """Focal classification loss plus smooth-L1 box loss, normalized by the positive count.

    The classification term is summed over every anchor. The box term only
    covers positive anchors (background anchors have no box target). Both are
    divided by the number of positive anchors in the batch (minimum 1), so an
    all-negative batch of "normal" images still yields a finite loss.

    Args:
        alpha : focal-loss weight on positives
        gamma : focal-loss focusing exponent (down-weights easy negatives)
        beta  : smooth-L1 transition point between L2 and L1
    """

    def __init__(self, alpha=0.25, gamma=1.5, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, cls_logits, box_pred, cls_targets, box_targets, positive_mask):
        """Return the scalar loss. cls_targets must be in [0, 1] (see train.build_batch_targets)."""
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