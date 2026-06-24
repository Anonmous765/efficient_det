"""
COCO mAP evaluation for a trained EfficientDet checkpoint.

Usage:
    python evaluate.py \
        --val-images   coco/images/val2017 \
        --val-ann      coco/annotations/instances_val2017.json \
        --checkpoint   checkpoints/best.pth \
        --phi 0 \
        --batch-size 8 \
        --workers 4
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from efficientdet import EfficientDet, EfficientDetConfig
from efficientdet.utils.nms import apply_nms

from dataset import CocoDataset, collate_fn
from dataset.transforms import Compose, Resize, ToTensor


def build_val_transforms(input_size: int):
    return Compose([Resize(input_size), ToTensor()])


@torch.no_grad()
def run_evaluation(model, loader, dataset, device, score_thresh=0.05, iou_thresh=0.5):
    model.eval()
    coco_results = []

    for images, _, _, img_ids in loader:
        images = images.to(device)
        class_preds, box_preds, anchors = model(images)
        detections = apply_nms(
            class_preds, box_preds, anchors,
            score_thresh=score_thresh, iou_thresh=iou_thresh,
        )

        for det, img_id in zip(detections, img_ids):
            if det["boxes"].numel() == 0:
                continue
            boxes  = det["boxes"].cpu()   # (K, 4) xyxy
            scores = det["scores"].cpu()  # (K,)
            labels = det["labels"].cpu()  # (K,)

            # COCO expects (x1, y1, w, h)
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]

            for k in range(boxes.shape[0]):
                coco_results.append({
                    "image_id":   int(img_id),
                    "category_id": dataset.label_to_cat_id[int(labels[k])],
                    "bbox":       boxes_xywh[k].tolist(),
                    "score":      float(scores[k]),
                })

    return coco_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-images",  default="coco/images/val2017")
    p.add_argument("--val-ann",     default="coco/annotations/instances_val2017.json")
    p.add_argument("--checkpoint",  default="checkpoints/best.pth")
    p.add_argument("--phi",         type=int,   default=0)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--score-thresh",type=float, default=0.05)
    p.add_argument("--iou-thresh",  type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EfficientDetConfig(phi=args.phi)

    val_ds = CocoDataset(
        args.val_images, args.val_ann,
        transforms=build_val_transforms(config.input_resolution),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )

    num_classes = val_ds.get_num_classes()
    model = EfficientDet(config, num_classes=num_classes).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    print("Running inference...")
    coco_results = run_evaluation(
        model, val_loader, val_ds, device,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    if not coco_results:
        print("No detections above score threshold — check your checkpoint or threshold.")
        return

    # Write results to a temp file and run COCOeval
    result_path = "coco_det_results.json"
    with open(result_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = val_ds.coco.loadRes(result_path)
    coco_eval = COCOeval(val_ds.coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.remove(result_path)


if __name__ == "__main__":
    main()
