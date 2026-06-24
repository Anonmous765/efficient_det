# efficientdet/utils/matcher.py
import torch
from efficientdet.utils.box_ops import cxcywh_to_xyxy, encode_boxes


def box_iou(a, b):
    """
    a : (N, 4) xyxy
    b : (M, 4) xyxy
    returns : (N, M) IoU matrix
    """
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # (M,)

    # broadcast: a[:, None, :] is (N,1,4), b[None] is (1,M,4)
    lt    = torch.max(a[:, None, :2], b[None, :, :2])     # (N, M, 2) top-left of overlap
    rb    = torch.min(a[:, None, 2:], b[None, :, 2:])     # (N, M, 2) bottom-right of overlap
    wh    = (rb - lt).clamp(min=0)                        # negative → no overlap → 0
    inter = wh[..., 0] * wh[..., 1]                       # (N, M)
    union = area_a[:, None] + area_b[None, :] - inter     # (N, M)

    return inter / union.clamp(min=1e-7)


@torch.no_grad()   # matching produces targets — no gradients needed
def match_anchors(anchors, gt_boxes, gt_labels,
                  num_classes, pos_thresh=0.5, neg_thresh=0.4):
    """
    anchors   : (N, 4)  cxcywh
    gt_boxes  : (M, 4)  cxcywh
    gt_labels : (M,)    long, 0-indexed class ids

    returns:
        positive_mask : (N,)    bool
        ignore_mask   : (N,)    bool
        cls_targets   : (N, C)  float, one-hot
        box_targets   : (N, 4)  float, encoded deltas
    """
    N = anchors.shape[0]
    device = anchors.device

    # --- handle empty image (no GT objects) ---
    if gt_boxes.numel() == 0:
        return (
            torch.zeros(N, dtype=torch.bool,    device=device),   # positive_mask
            torch.zeros(N, dtype=torch.bool,    device=device),   # ignore_mask
            torch.zeros(N, num_classes,         device=device),   # cls_targets
            torch.zeros(N, 4,                   device=device),   # box_targets
        )

    # --- Step 1: compute (N, M) IoU matrix ---
    # IoU needs xyxy format — convert both
    anchors_xyxy  = cxcywh_to_xyxy(anchors)   # (N, 4)
    gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes)  # (M, 4)
    iou = box_iou(anchors_xyxy, gt_boxes_xyxy) # (N, M)

    # --- Step 2: for each anchor, find its best-matching GT ---
    best_iou, best_gt_idx = iou.max(dim=1)     # both (N,)

    # --- Step 3: threshold into pos / neg / ignore ---
    pos_mask    = best_iou >= pos_thresh        # (N,) bool
    neg_mask    = best_iou <  neg_thresh        # (N,) bool
    ignore_mask = ~pos_mask & ~neg_mask         # (N,) bool

    # --- Step 4: build cls_targets (N, C) ---
    cls_targets = torch.zeros(N, num_classes, device=device)
    if pos_mask.any():
        pos_cls = gt_labels[best_gt_idx[pos_mask]]   # class id for each positive
        cls_targets[pos_mask, pos_cls] = 1.0

    # zero out ignored anchors so they don't contribute to focal loss
    cls_targets[ignore_mask] = -1.0   # sentinel — we'll mask these in the loss

    # --- Step 5: build box_targets (N, 4) ---
    box_targets = torch.zeros(N, 4, device=device)
    if pos_mask.any():
        matched_gt = gt_boxes[best_gt_idx[pos_mask]]        # (num_pos, 4) cxcywh
        box_targets[pos_mask] = encode_boxes(matched_gt, anchors[pos_mask])

    return pos_mask, ignore_mask, cls_targets, box_targets