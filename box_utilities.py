import torch

def encode_boxes(gt_boxes, anchors):
    """
    gt_boxes : (N, 4) ground-truth boxes in (cx, cy, w, h)
    anchors  : (N, 4) matched anchors   in (cx, cy, w, h)
    Returns  : (N, 4) delta targets [tx, ty, tw, th]
    Used during: training (building regression targets for the loss)
    """
    t_x = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    t_y = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    t_w = torch.log(gt_boxes[:, 2] / anchors[:, 2])
    t_h = torch.log(gt_boxes[:, 3] / anchors[:, 3])
    return torch.stack([t_x, t_y, t_w, t_h], dim=-1)


def decode_boxes(deltas, anchors):
    """
    deltas  : (B, N, 4) predicted deltas from box head
    anchors : (N, 4)    anchor boxes in (cx, cy, w, h)
    Returns : (B, N, 4) decoded boxes in (cx, cy, w, h)
    Used during: inference and validation (converting predictions to real boxes)
    """
    a = anchors[None]   # (1, N, 4) — broadcasts over batch dim
    pred_cx = deltas[..., 0] * a[..., 2] + a[..., 0]
    pred_cy = deltas[..., 1] * a[..., 3] + a[..., 1]
    pred_w  = a[..., 2] * torch.exp(deltas[..., 2])
    pred_h  = a[..., 3] * torch.exp(deltas[..., 3])
    return torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)


def cxcywh_to_xyxy(boxes):
    """(cx, cy, w, h) → (x1, y1, x2, y2) — needed for IoU computation"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2,
                        cx + w/2, cy + h/2], dim=-1)


def xyxy_to_cxcywh(boxes):
    """(x1, y1, x2, y2) → (cx, cy, w, h) — needed after loading GT annotations"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2,
                        x2-x1,     y2-y1], dim=-1)