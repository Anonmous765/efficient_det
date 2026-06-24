"""
EfficientDet training on MS COCO 2017.

Usage:
    python train.py \
        --train-images coco/images/train2017 \
        --train-ann    coco/annotations/instances_train2017.json \
        --val-images   coco/images/val2017 \
        --val-ann      coco/annotations/instances_val2017.json \
        --phi 0 \
        --epochs 300 \
        --batch-size 8 \
        --lr 1e-4 \
        --workers 4 \
        --checkpoint-dir checkpoints
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from efficientdet import EfficientDet, EfficientDetConfig
from efficientdet.utils.loss import EfficientDetLoss
from efficientdet.utils.matcher import match_anchors

from dataset import CocoDataset, collate_fn
from dataset.transforms import Compose, Resize, RandomHorizontalFlip, ColorJitter, ToTensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_transforms(input_size: int, train: bool):
    if train:
        return Compose([
            Resize(input_size),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToTensor(),
        ])
    return Compose([Resize(input_size), ToTensor()])


def build_batch_targets(gt_boxes_list, gt_labels_list, anchors, num_classes, device):
    """Call match_anchors per image and stack into batch tensors.

    match_anchors marks ignored anchors (ambiguous IoU zone) with -1 in
    cls_targets as a sentinel. sigmoid_focal_loss expects [0,1] targets, so
    we clamp to 0 here — ignored anchors are treated as negatives rather
    than excluded entirely, which is a conservative but numerically safe
    approximation.
    """
    cls_targets_list  = []
    box_targets_list  = []
    pos_mask_list     = []

    for gt_boxes, gt_labels in zip(gt_boxes_list, gt_labels_list):
        gt_boxes  = gt_boxes.to(device)
        gt_labels = gt_labels.to(device)

        pos_mask, _, cls_targets, box_targets = match_anchors(
            anchors, gt_boxes, gt_labels, num_classes
        )
        # Clamp out the -1 ignore sentinel before feeding to focal loss
        cls_targets = cls_targets.clamp(min=0.0)
        cls_targets_list.append(cls_targets)
        box_targets_list.append(box_targets)
        pos_mask_list.append(pos_mask)

    return (
        torch.stack(cls_targets_list),   # (B, N, C)
        torch.stack(box_targets_list),   # (B, N, 4)
        torch.stack(pos_mask_list),      # (B, N)
    )


def save_checkpoint(path, epoch, model, optimizer, val_loss):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss":        val_loss,
    }, path)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, gt_boxes_list, gt_labels_list, _) in enumerate(loader):
        images = images.to(device)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            class_preds, box_preds, anchors = model(images)
            cls_targets, box_targets, pos_mask = build_batch_targets(
                gt_boxes_list, gt_labels_list, anchors, model.num_classes, device
            )
            loss = criterion(class_preds, box_preds, cls_targets, box_targets, pos_mask)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  step {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for images, gt_boxes_list, gt_labels_list, _ in loader:
        images = images.to(device)
        class_preds, box_preds, anchors = model(images)
        cls_targets, box_targets, pos_mask = build_batch_targets(
            gt_boxes_list, gt_labels_list, anchors, model.num_classes, device
        )
        loss = criterion(class_preds, box_preds, cls_targets, box_targets, pos_mask)
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images",   default="coco/images/train2017")
    p.add_argument("--train-ann",      default="coco/annotations/instances_train2017.json")
    p.add_argument("--val-images",     default="coco/images/val2017")
    p.add_argument("--val-ann",        default="coco/annotations/instances_val2017.json")
    p.add_argument("--phi",            type=int,   default=0)
    p.add_argument("--epochs",         type=int,   default=300)
    p.add_argument("--batch-size",     type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume",         default=None, help="path to checkpoint to resume from")
    p.add_argument("--no-amp",         action="store_true", help="disable mixed-precision training")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EfficientDetConfig(phi=args.phi)

    # Datasets
    train_ds = CocoDataset(
        args.train_images, args.train_ann,
        transforms=build_transforms(config.input_resolution, train=True),
    )
    val_ds = CocoDataset(
        args.val_images, args.val_ann,
        transforms=build_transforms(config.input_resolution, train=False),
    )
    num_classes = train_ds.get_num_classes()
    print(f"Classes: {num_classes}  |  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )

    # Model
    model = EfficientDet(config, num_classes=num_classes).to(device)
    criterion = EfficientDetLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if (not args.no_amp and device.type == "cuda") else None

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Save latest
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "last.pth"),
            epoch, model, optimizer, val_loss,
        )
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pth"),
                epoch, model, optimizer, val_loss,
            )
            print(f"  ** New best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
