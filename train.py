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
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

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


def plot_loss_history(history, out_path):
    """Plot train vs val loss per epoch — diverging curves signal overfitting."""
    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train"], label="train loss", marker="o", ms=3)
    plt.plot(epochs, history["val"],   label="val loss",   marker="o", ms=3)
    best = min(range(len(history["val"])), key=lambda i: history["val"][i])
    plt.axvline(best + 1, color="gray", ls="--", alpha=0.5,
                label=f"best val (epoch {best + 1})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, amp, num_classes):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, gt_boxes_list, gt_labels_list, _) in enumerate(loader):
        images = images.to(device, memory_format=torch.channels_last)

        # bf16 autocast on Blackwell: full dynamic range, so no GradScaler needed
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            class_preds, box_preds, anchors = model(images)
            cls_targets, box_targets, pos_mask = build_batch_targets(
                gt_boxes_list, gt_labels_list, anchors, num_classes, device
            )
            loss = criterion(class_preds, box_preds, cls_targets, box_targets, pos_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  step {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, amp, num_classes):
    model.eval()
    total_loss = 0.0

    for images, gt_boxes_list, gt_labels_list, _ in loader:
        images = images.to(device, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            class_preds, box_preds, anchors = model(images)
            cls_targets, box_targets, pos_mask = build_batch_targets(
                gt_boxes_list, gt_labels_list, anchors, num_classes, device
            )
            loss = criterion(class_preds, box_preds, cls_targets, box_targets, pos_mask)
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images",   default="coco2017/train2017")
    p.add_argument("--train-ann",      default="coco2017/annotations/instances_train2017.json")
    p.add_argument("--val-images",     default="coco2017/val2017")
    p.add_argument("--val-ann",        default="coco2017/annotations/instances_val2017.json")
    p.add_argument("--test-fraction",  type=float, default=0.05,
                   help="fraction of train2017 held out as the test split (must match evaluate.py)")
    p.add_argument("--split-seed",     type=int,   default=42,
                   help="seed controlling the train/test split (must match evaluate.py)")
    p.add_argument("--keep-empty",     action="store_true",
                   help="keep annotation-free images as negatives (needed when the "
                        "dataset includes 'normal' background images)")
    p.add_argument("--phi",            type=int,   default=0)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch-size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume",         default=None, help="path to checkpoint to resume from")
    p.add_argument("--no-amp",         action="store_true", help="disable bf16 mixed-precision training")
    p.add_argument("--compile",        action="store_true",
                   help="wrap the model in torch.compile (slow first step, faster afterwards)")
    p.add_argument("--patience",       type=int,   default=0,
                   help="early-stop after N epochs without val-loss improvement (0 = disabled)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # TF32 matmuls/convs + autotuned kernels for the fixed input size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    config = EfficientDetConfig(phi=args.phi)

    # Datasets
    # train2017 minus the held-out test slice (test images are never trained on)
    train_ds = CocoDataset(
        args.train_images, args.train_ann,
        transforms=build_transforms(config.input_resolution, train=True),
        split="train", test_fraction=args.test_fraction, seed=args.split_seed,
        keep_empty=args.keep_empty,
    )
    # val2017 is the validation set, used in full
    val_ds = CocoDataset(
        args.val_images, args.val_ann,
        transforms=build_transforms(config.input_resolution, train=False),
        keep_empty=args.keep_empty,
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
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)  # better Tensor-Core use
    criterion = EfficientDetLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    amp = (not args.no_amp) and device.type == "cuda"

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_since_improve = 0
    history = {"train": [], "val": []}

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history_path = os.path.join(args.checkpoint_dir, "loss_history.json")
    curve_path   = os.path.join(args.checkpoint_dir, "loss_curve.png")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}")
        # Carry forward the loss history so the curve continues unbroken
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)

    # Compile AFTER loading weights; checkpoints still save the original `model`
    # so its state_dict keys stay clean (no _orig_mod. prefix) for evaluate.py.
    train_model = torch.compile(model) if args.compile else model

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(train_model, train_loader, criterion, optimizer, device, amp, num_classes)
        val_loss   = validate(train_model, val_loader, criterion, device, amp, num_classes)
        scheduler.step()

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Record history and refresh the loss curve (so it survives an interrupt)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        plot_loss_history(history, curve_path)

        # Save latest
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "last.pth"),
            epoch, model, optimizer, val_loss,
        )
        # Save best + track early-stopping patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improve = 0
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pth"),
                epoch, model, optimizer, val_loss,
            )
            print(f"  ** New best val loss: {best_val_loss:.4f}")
        else:
            epochs_since_improve += 1
            if args.patience and epochs_since_improve >= args.patience:
                print(f"\nEarly stop: no val improvement for {args.patience} epochs "
                      f"(best={best_val_loss:.4f}).")
                break

    print(f"\nDone. Loss curve: {curve_path}  |  history: {history_path}")


if __name__ == "__main__":
    main()
