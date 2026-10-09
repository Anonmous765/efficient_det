"""
Lay out data/anomaly in the folder structure anomalib's ``Folder`` datamodule expects.

data/anomaly is a flat image folder plus COCO JSONs, where an image with zero
boxes is a Normal (defect-free) print. anomalib instead wants one directory per
class, so this script symlinks each image into

    <out-dir>/train/good/        normal images (the only training data)
    <out-dir>/{val,test}/good/   normal images
    <out-dir>/{val,test}/defect/ anomalous images
    <out-dir>/{val,test}/mask/   binary PNG masks for the defect images

EfficientAd is one-class: it learns from normal images only, so the defect
images of the COCO train split are useless for training. Instead they are
moved into val and test, which otherwise hold only ~60 defects each - too few
for a stable threshold or trustworthy metrics. Because of this the val/test
splits no longer match the ones EfficientDet is evaluated on.

The COCO train split is a Roboflow export with several augmented copies of
each source image (``<source>.rf.<hash>.jpg``). Moved defects are therefore
handled per source image: one copy is kept, sources already present in val or
test are dropped, and the rest are shuffled (fixed seed) and divided between
val and test, so no source image ends up on both sides.

The masks are the COCO boxes filled in as white rectangles, so pixel-level
metrics are only a rough guide: a box covers more than the defect itself.

The split folders under --out-dir are deleted and rebuilt on every run.

Usage (from the repo root):
    python -m efficientAD.prepare_folders
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict

from PIL import Image, ImageDraw

SPLITS = ("train", "val", "test")


def source_name(file_name):
    """Strip Roboflow's ``.rf.<hash>.<ext>`` suffix, leaving the original image's name."""
    return re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", file_name)


def load_split(split, ann_dir):
    """Read one split's COCO JSON into (good, defect) lists; defect items carry their boxes."""
    with open(os.path.join(ann_dir, f"instances_{split}.json")) as f:
        coco = json.load(f)

    boxes = defaultdict(list)
    for ann in coco["annotations"]:
        boxes[ann["image_id"]].append(ann["bbox"])

    good, defect = [], []
    for img in coco["images"]:
        img_boxes = boxes[img["id"]]
        if img_boxes:
            defect.append({**img, "boxes": img_boxes})
        else:
            good.append(img)
    return good, defect


def split_train_defects(train_defect, held_out, val_fraction, seed):
    """Divide the train split's defects between val and test, one copy per source image.

    Sources whose name is in ``held_out`` (already in val or test) are dropped.
    """
    by_source = {}
    for img in sorted(train_defect, key=lambda i: i["file_name"]):
        by_source.setdefault(source_name(img["file_name"]), img)
    sources = sorted(s for s in by_source if s not in held_out)
    random.Random(seed).shuffle(sources)

    n_val = round(len(sources) * val_fraction)
    to_val = [by_source[s] for s in sources[:n_val]]
    to_test = [by_source[s] for s in sources[n_val:]]
    print(f"train defects: {len(train_defect)} images, {len(by_source)} sources, "
          f"{len(by_source) - len(sources)} dropped (already in val/test), "
          f"{len(to_val)} -> val, {len(to_test)} -> test")
    return to_val, to_test


def write_split(split, good, defect, images_dir, out_dir):
    """Symlink one split's images into good/ (and defect/) and draw its box masks."""
    split_dir = os.path.join(out_dir, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)

    def link(img, folder):
        os.makedirs(folder, exist_ok=True)
        src = os.path.abspath(os.path.join(images_dir, img["file_name"]))
        os.symlink(src, os.path.join(folder, img["file_name"]))

    for img in good:
        link(img, os.path.join(split_dir, "good"))

    mask_dir = os.path.join(split_dir, "mask")
    for img in defect:
        link(img, os.path.join(split_dir, "defect"))
        os.makedirs(mask_dir, exist_ok=True)
        # Same stem as the image: anomalib pairs masks to images by sorted name.
        stem = os.path.splitext(img["file_name"])[0]
        mask = Image.new("L", (img["width"], img["height"]), 0)
        draw = ImageDraw.Draw(mask)
        for x, y, w, h in img["boxes"]:
            draw.rectangle([x, y, x + w, y + h], fill=255)
        mask.save(os.path.join(mask_dir, stem + ".png"))

    print(f"{split}: {len(good)} good, {len(defect)} defect")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="data/anomaly/images")
    parser.add_argument("--ann-dir", default="data/anomaly/annotations")
    parser.add_argument("--out-dir", default="data/anomaly_folder")
    parser.add_argument("--val-fraction", type=float, default=0.5,
                        help="share of the moved train defects that goes to val (rest to test)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = {split: load_split(split, args.ann_dir) for split in SPLITS}

    held_out = {source_name(img["file_name"])
                for split in ("val", "test") for imgs in data[split] for img in imgs}
    to_val, to_test = split_train_defects(data["train"][1], held_out, args.val_fraction, args.seed)

    write_split("train", data["train"][0], [], args.images_dir, args.out_dir)
    write_split("val", data["val"][0], data["val"][1] + to_val, args.images_dir, args.out_dir)
    write_split("test", data["test"][0], data["test"][1] + to_test, args.images_dir, args.out_dir)


if __name__ == "__main__":
    main()
