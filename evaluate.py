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
    """Run inference over the loader.

    Returns (coco_results, img_top_score, img_has_gt):
      coco_results  : detections in COCO submission format, for COCOeval
      img_top_score : {image_id: highest detection score, 0.0 if none}
      img_has_gt    : {image_id: image has at least one ground-truth box}

    The last two drive the image-level confusion matrix. Collecting them here
    means the threshold sweep re-scores cached numbers instead of re-running
    inference once per candidate threshold.
    """
    model.eval()
    coco_results = []
    img_top_score = {}
    img_has_gt    = {}

    for images, gt_boxes_list, _, img_ids in loader:
        images = images.to(device)
        class_preds, box_preds, anchors = model(images)
        detections = apply_nms(
            class_preds, box_preds, anchors,
            score_thresh=score_thresh, iou_thresh=iou_thresh,
        )

        for det, gt_boxes, img_id in zip(detections, gt_boxes_list, img_ids):
            img_id = int(img_id)
            img_has_gt[img_id] = bool(gt_boxes.shape[0] > 0)

            if det["boxes"].numel() == 0:
                img_top_score[img_id] = 0.0
                continue
            boxes  = det["boxes"].cpu()   # (K, 4) xyxy
            scores = det["scores"].cpu()  # (K,)
            labels = det["labels"].cpu()  # (K,)

            img_top_score[img_id] = float(scores.max())

            # COCO expects (x1, y1, w, h)
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]

            for k in range(boxes.shape[0]):
                coco_results.append({
                    "image_id":   img_id,
                    "category_id": dataset.label_to_cat_id[int(labels[k])],
                    "bbox":       boxes_xywh[k].tolist(),
                    "score":      float(scores[k]),
                })

    return coco_results, img_top_score, img_has_gt


# ---------------------------------------------------------------------------
# Image-level ("is this print defective?") metrics
# ---------------------------------------------------------------------------

def confusion_at(img_top_score, img_has_gt, thresh):
    """Count TP/FP/FN/TN treating each image as one binary decision.

    An image is *predicted* anomalous when its highest-scoring detection
    clears `thresh`, and is *actually* anomalous when it carries at least one
    ground-truth box. Normal images (deliberately annotation-free) are the
    negatives, so a detection on one is a false positive.
    """
    tp = fp = fn = tn = 0
    for img_id, has_gt in img_has_gt.items():
        pred_pos = img_top_score.get(img_id, 0.0) >= thresh
        if   has_gt and pred_pos:         tp += 1
        elif has_gt and not pred_pos:     fn += 1
        elif not has_gt and pred_pos:     fp += 1
        else:                             tn += 1
    return tp, fp, fn, tn


def derive_metrics(tp, fp, fn, tn):
    """accuracy, precision, recall, specificity, F1 — 0.0 where undefined."""
    total = tp + fp + fn + tn
    acc  = (tp + tn) / total     if total       else 0.0
    prec = tp / (tp + fp)        if (tp + fp)   else 0.0
    rec  = tp / (tp + fn)        if (tp + fn)   else 0.0
    spec = tn / (tn + fp)        if (tn + fp)   else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, spec, f1


def print_image_level_report(img_top_score, img_has_gt, chosen_thresh=None):
    n_pos = sum(1 for v in img_has_gt.values() if v)
    n_neg = len(img_has_gt) - n_pos

    print("\n" + "=" * 66)
    print("IMAGE-LEVEL DETECTION  (does this image contain an anomaly?)")
    print("=" * 66)
    print(f"{len(img_has_gt)} images: {n_pos} anomaly (has boxes) / "
          f"{n_neg} normal (no boxes)")

    if n_neg == 0:
        print("\nNOTE: no annotation-free images in this split, so there are no\n"
              "      true negatives — specificity and false positives are\n"
              "      meaningless here. Re-run with --keep-empty.")

    # Sweep the decision threshold over the cached scores.
    sweep = [round(0.05 * i, 2) for i in range(1, 20)]
    print(f"\n{'thresh':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
          f"{'acc':>7} {'prec':>7} {'recall':>7} {'spec':>7} {'F1':>7}")
    print("-" * 66)
    best_f1, best_t = -1.0, sweep[0]
    for t in sweep:
        tp, fp, fn, tn = confusion_at(img_top_score, img_has_gt, t)
        acc, prec, rec, spec, f1 = derive_metrics(tp, fp, fn, tn)
        if f1 > best_f1:
            best_f1, best_t = f1, t
        print(f"{t:>7.2f} {tp:>5} {fp:>5} {fn:>5} {tn:>5} "
              f"{acc:>7.3f} {prec:>7.3f} {rec:>7.3f} {spec:>7.3f} {f1:>7.3f}")

    thresh = chosen_thresh if chosen_thresh is not None else best_t
    label  = "requested" if chosen_thresh is not None else "best-F1"
    tp, fp, fn, tn = confusion_at(img_top_score, img_has_gt, thresh)
    acc, prec, rec, spec, f1 = derive_metrics(tp, fp, fn, tn)

    print(f"\nConfusion matrix @ score >= {thresh:.2f}  ({label} threshold)")
    print("                     predicted")
    print("                 anomaly   normal")
    print(f"  actual anomaly {tp:>7} {fn:>8}")
    print(f"  actual normal  {fp:>7} {tn:>8}")
    print(f"\n  accuracy    {acc:.4f}   ({tp + tn}/{tp + fp + fn + tn} images correct)")
    print(f"  precision   {prec:.4f}   (of images flagged, this fraction really was defective)")
    print(f"  recall      {rec:.4f}   (of defective prints, this fraction was caught)")
    print(f"  specificity {spec:.4f}   (of good prints, this fraction was left alone)")
    print(f"  F1          {f1:.4f}")
    print(f"\n  {fn} missed defect(s), {fp} false alarm(s)")


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
    p.add_argument("--decision-thresh", type=float, default=None,
                   help="score threshold for the image-level confusion matrix "
                        "(an image is flagged anomalous when its best detection "
                        "clears it). Defaults to whichever swept threshold "
                        "maximizes F1.")
    p.add_argument("--no-image-level", action="store_true",
                   help="skip the image-level confusion matrix and report only COCO mAP")
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
    coco_results, img_top_score, img_has_gt = run_evaluation(
        model, val_loader, val_ds, device,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    # Report image-level results first: "the model detected nothing" is a real
    # (bad) outcome the confusion matrix should show, not a reason to bail out.
    if not args.no_image_level:
        print_image_level_report(img_top_score, img_has_gt, args.decision_thresh)

    if not coco_results:
        print("\nNo detections above score threshold — skipping COCO mAP. "
              "Check your checkpoint, or lower --score-thresh.")
        return

    if not args.no_image_level:
        print("\n" + "=" * 66)
        print("BOX-LEVEL LOCALIZATION  (COCO mAP)")
        print("=" * 66)

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
