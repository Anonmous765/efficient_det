"""
Custom collate function for variable-length detection annotations.

Each image in a batch can have a different number of ground-truth boxes,
so we cannot stack them into a single tensor.  Instead we return lists.

Returns:
    images    : FloatTensor[B, 3, H, W]
    gt_boxes  : list of FloatTensor[Mi, 4]  — one per image, (cx, cy, w, h)
    gt_labels : list of LongTensor[Mi]       — one per image
    img_ids   : list of int
"""
import torch


def collate_fn(batch):
    images, gt_boxes, gt_labels, img_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(gt_boxes), list(gt_labels), list(img_ids)
