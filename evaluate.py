"""
COCO mAP evaluation (the "test" stage) for a trained EfficientDet checkpoint.

By default this evaluates on the held-out test split carved out of train2017 —
the same slice train.py excludes from training. test_fraction / split_seed must
match the values used during training for the split to line up.

Usage:
    python evaluate.py \
        --images       coco2017/train2017 \
        --ann          coco2017/annotations/instances_train2017.json \
        --split        test \
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
    p.add_argument("--images",      default="coco2017/train2017")
    p.add_argument("--ann",         default="coco2017/annotations/instances_train2017.json")
    p.add_argument("--split",       default="test", choices=["test", "train", "all"],
                   help="which split to evaluate ('all' = every annotated image)")
    p.add_argument("--test-fraction", type=float, default=0.05,
                   help="held-out test fraction (must match train.py)")
    p.add_argument("--split-seed",  type=int,   default=42,
                   help="split seed (must match train.py)")
    p.add_argument("--checkpoint",  default="checkpoints/best.pth")
    p.add_argument("--phi",         type=int,   default=0)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--score-thresh",type=float, default=0.05)
    p.add_argument("--iou-thresh",  type=float, default=0.5)
    p.add_argument("--keep-empty",  action="store_true",
                   help="keep annotation-free images so detections on 'normal' "
                        "images count as false positives (must match train.py)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EfficientDetConfig(phi=args.phi)

    split = None if args.split == "all" else args.split
    val_ds = CocoDataset(
        args.images, args.ann,
        transforms=build_val_transforms(config.input_resolution),
        split=split, test_fraction=args.test_fraction, seed=args.split_seed,
        keep_empty=args.keep_empty,
    )
    print(f"Evaluating split='{args.split}'  |  {len(val_ds)} images")
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
    # Restrict scoring to the evaluated split; otherwise the annotation file's
    # other images (e.g. the ~112k train images) count as missed detections.
    coco_eval.params.imgIds = sorted(val_ds.ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.remove(result_path)


if __name__ == "__main__":
    main()
