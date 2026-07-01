"""
Visualize EfficientDet predictions: draw predicted boxes (and, by default, the
ground-truth boxes for comparison) on a handful of images and save them as PNGs.

Predicted boxes are drawn in red with their class name + score; ground-truth
boxes are drawn in green. By default it pulls from the held-out test split of
train2017 — the same slice train.py never trains on.

Usage:
    python visualize.py \
        --checkpoint checkpoints/best.pth \
        --images     coco2017/train2017 \
        --ann        coco2017/annotations/instances_train2017.json \
        --split      test \
        --num-images 12 \
        --score-thresh 0.3 \
        --out-dir    predictions
"""
import argparse
import os

import torch
from PIL import Image, ImageDraw

from efficientdet import EfficientDet, EfficientDetConfig
from efficientdet.utils.nms import apply_nms
from efficientdet.utils.box_ops import cxcywh_to_xyxy

from dataset import CocoDataset
from dataset.transforms import Compose, Resize, ToTensor


# ImageNet stats used by dataset.transforms.ToTensor — needed to invert
# normalisation back to a viewable RGB image.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Undo ImageNet normalisation and convert (3, H, W) float → PIL RGB."""
    img = image.cpu() * _STD + _MEAN
    img = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def draw_boxes(draw, boxes, labels, label_to_name, color, scores=None):
    """Draw xyxy boxes with a class label (and optional score) onto an image."""
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = label_to_name.get(int(labels[i]), str(int(labels[i])))
        text = f"{name} {scores[i]:.2f}" if scores is not None else name
        # Small filled caption box above the bbox for legibility
        ty = max(0, y1 - 11)
        draw.rectangle([x1, ty, x1 + 6.5 * len(text), ty + 11], fill=color)
        draw.text((x1 + 1, ty), text, fill="white")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   default="checkpoints/best.pth")
    p.add_argument("--images",       default="coco2017/train2017")
    p.add_argument("--ann",          default="coco2017/annotations/instances_train2017.json")
    p.add_argument("--split",        default="test", choices=["test", "train", "all"],
                   help="which split to sample from ('all' = every annotated image)")
    p.add_argument("--test-fraction", type=float, default=0.05,
                   help="held-out test fraction (must match train.py)")
    p.add_argument("--split-seed",   type=int,   default=42,
                   help="split seed (must match train.py)")
    p.add_argument("--phi",          type=int,   default=0)
    p.add_argument("--num-images",   type=int,   default=12)
    p.add_argument("--score-thresh", type=float, default=0.3)
    p.add_argument("--iou-thresh",   type=float, default=0.5)
    p.add_argument("--no-gt",        action="store_true",
                   help="don't draw ground-truth boxes (predictions only)")
    p.add_argument("--out-dir",      default="predictions")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EfficientDetConfig(phi=args.phi)
    split = None if args.split == "all" else args.split
    ds = CocoDataset(
        args.images, args.ann,
        transforms=Compose([Resize(config.input_resolution), ToTensor()]),
        split=split, test_fraction=args.test_fraction, seed=args.split_seed,
    )

    # 0-indexed label -> human-readable COCO category name
    cat_id_to_name = {c["id"]: c["name"] for c in ds.coco.loadCats(ds.coco.getCatIds())}
    label_to_name = {label: cat_id_to_name[cat_id]
                     for label, cat_id in ds.label_to_cat_id.items()}

    num_classes = ds.get_num_classes()
    model = EfficientDet(config, num_classes=num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  |  split='{args.split}'")

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.num_images, len(ds))

    for idx in range(n):
        image, gt_boxes, gt_labels, img_id = ds[idx]

        class_preds, box_preds, anchors = model(image.unsqueeze(0).to(device))
        det = apply_nms(
            class_preds, box_preds, anchors,
            score_thresh=args.score_thresh, iou_thresh=args.iou_thresh,
        )[0]

        pil = tensor_to_pil(image)
        draw = ImageDraw.Draw(pil)

        if not args.no_gt and gt_boxes.numel() > 0:
            gt_xyxy = cxcywh_to_xyxy(gt_boxes)
            draw_boxes(draw, gt_xyxy, gt_labels, label_to_name, color="lime")

        draw_boxes(
            draw, det["boxes"].cpu(), det["labels"].cpu(), label_to_name,
            color="red", scores=det["scores"].cpu(),
        )

        out_path = os.path.join(args.out_dir, f"{idx:03d}_img{int(img_id)}.png")
        pil.save(out_path)
        print(f"  {out_path}  ({det['boxes'].shape[0]} predictions)")

    print(f"\nSaved {n} images to {args.out_dir}/  (red = prediction, green = ground truth)")


if __name__ == "__main__":
    main()
