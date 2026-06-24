"""
Post-processing: decode raw model outputs → final detections.

apply_nms returns one dict per image in the batch:
    {
        'boxes'  : FloatTensor[K, 4]  (x1, y1, x2, y2) absolute pixels
        'scores' : FloatTensor[K]
        'labels' : LongTensor[K]      0-indexed class ids
    }
"""
import torch
from torchvision.ops import batched_nms

from efficientdet.utils.box_ops import decode_boxes, cxcywh_to_xyxy


def apply_nms(
    class_preds,
    box_preds,
    anchors,
    score_thresh: float = 0.05,
    iou_thresh:   float = 0.5,
    max_dets:     int   = 100,
):
    """
    class_preds : (B, N, num_classes)  raw logits (pre-sigmoid)
    box_preds   : (B, N, 4)            raw deltas
    anchors     : (N, 4)               (cx, cy, w, h) absolute pixels
    """
    B = class_preds.shape[0]
    scores_all = class_preds.sigmoid()  # (B, N, C)
    boxes_all  = cxcywh_to_xyxy(decode_boxes(box_preds, anchors))  # (B, N, 4)

    results = []
    for i in range(B):
        scores_i = scores_all[i]   # (N, C)
        boxes_i  = boxes_all[i]    # (N, 4)

        # Flatten across classes: each (anchor, class) pair is a candidate
        scores_flat, labels_flat = scores_i.max(dim=1)  # (N,)

        keep = scores_flat > score_thresh
        scores_flat = scores_flat[keep]
        labels_flat = labels_flat[keep]
        boxes_flat  = boxes_i[keep]

        if scores_flat.numel() == 0:
            results.append({
                "boxes":  torch.zeros((0, 4), device=class_preds.device),
                "scores": torch.zeros((0,),   device=class_preds.device),
                "labels": torch.zeros((0,),   device=class_preds.device, dtype=torch.int64),
            })
            continue

        keep_idx = batched_nms(boxes_flat, scores_flat, labels_flat, iou_thresh)
        keep_idx = keep_idx[:max_dets]

        results.append({
            "boxes":  boxes_flat[keep_idx],
            "scores": scores_flat[keep_idx],
            "labels": labels_flat[keep_idx],
        })

    return results
