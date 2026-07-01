"""
Prepare the custom 3D-print anomaly dataset for the EfficientDet pipeline.

Two sources are combined:

  * ``result.json``  -- the Label Studio COCO export holding the *bounding boxes*
    for the anomaly images (the source of truth for annotations).
  * ``Full_dataset/training_data`` -- a train/val/test split laid out as
    ``<split>/Anomaly`` and ``<split>/Normal`` folders. The ``Anomaly`` folders
    hold the annotated images; the ``Normal`` folders hold defect-free prints
    that we add as *negatives* (image entries with zero boxes) so the detector
    learns to not fire on clean parts.

The script:
  1. reads the predefined train/val/test split from the Full_dataset folders
     (test > val > train priority so eval sets stay clean of duplicates),
  2. for every annotated image, attaches its boxes from ``result.json`` and
     copies the file into a flat ``<out-dir>/images`` folder,
  3. adds every ``Normal`` image as a zero-annotation negative,
  4. remaps the ``Anomaly`` category id 0 -> 1 (standard 1-indexed COCO),
  5. writes ``instances_{train,val,test}.json`` under ``<out-dir>/annotations``.

Then train / evaluate with the ``--keep-empty`` flag so the negatives are used:

    python train.py \
        --train-images data/anomaly/images \
        --train-ann    data/anomaly/annotations/instances_train.json \
        --val-images   data/anomaly/images \
        --val-ann      data/anomaly/annotations/instances_val.json \
        --phi 0 --test-fraction 0 --keep-empty
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

from PIL import Image

SPLITS = ("train", "val", "test")
# Folders searched (in order) for the handful of annotated images that are not
# present under Full_dataset/<split>/Anomaly.
FALLBACK_SUBDIRS = ("default", "train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result-json", default="result.json",
                   help="Label Studio COCO export with the anomaly boxes")
    p.add_argument("--full-dataset",
                   default=os.path.expanduser(
                       "~/Desktop/images/Full_dataset/training_data"),
                   help="root holding <split>/Anomaly and <split>/Normal folders")
    p.add_argument("--fallback-root",
                   default=os.path.expanduser("~/Desktop/images/coco/images"),
                   help="where to find annotated images missing from Full_dataset")
    p.add_argument("--out-dir", default="data/anomaly",
                   help="destination for the flattened images + split JSONs")
    return p.parse_args()


def scan_split_folders(full_dataset):
    """Return {basename: (split, abs_path)} for Anomaly and Normal folders.

    Iterating train -> val -> test means a basename present in more than one
    split ends up assigned to the *last* one seen (test), keeping the eval
    splits authoritative and free of train duplicates.
    """
    anomaly, normal = {}, {}
    for split in SPLITS:
        for kind, table in (("Anomaly", anomaly), ("Normal", normal)):
            folder = os.path.join(full_dataset, split, kind)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                table[name] = (split, os.path.join(folder, name))
    return anomaly, normal


def find_fallback(basename, fallback_root):
    for sub in FALLBACK_SUBDIRS:
        cand = os.path.join(fallback_root, sub, basename)
        if os.path.exists(cand):
            return cand
    return ""


def main():
    args = parse_args()

    with open(args.result_json) as f:
        raw = json.load(f)

    # --- Index the annotations by image basename ----------------------------
    id_to_bn = {im["id"]: os.path.basename(im["file_name"]) for im in raw["images"]}
    bn_to_size = {os.path.basename(im["file_name"]): (im["width"], im["height"])
                  for im in raw["images"]}
    bn_to_anns = defaultdict(list)
    for ann in raw["annotations"]:
        bn_to_anns[id_to_bn[ann["image_id"]]].append(ann)
    annotated = sorted(bn_to_anns.keys())
    print(f"Loaded {len(annotated)} annotated images / "
          f"{len(raw['annotations'])} boxes from {args.result_json}")

    # --- Read the predefined split from the Full_dataset folders ------------
    anomaly_loc, normal_loc = scan_split_folders(args.full_dataset)
    print(f"Full_dataset: {len(anomaly_loc)} Anomaly files, "
          f"{len(normal_loc)} Normal files (deduped across splits)")

    img_dir = os.path.join(args.out_dir, "images")
    ann_dir = os.path.join(args.out_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Accumulators, keyed by split
    images_by_split = {s: [] for s in SPLITS}
    anns_by_split = {s: [] for s in SPLITS}
    next_img_id = 0
    next_ann_id = 0
    copied = 0
    fallback_used = 0

    def copy_in(src, basename):
        nonlocal copied
        dst = os.path.join(img_dir, basename)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1

    # --- Positives: annotated anomaly images --------------------------------
    for bn in annotated:
        if bn in anomaly_loc:
            split, src = anomaly_loc[bn]
        else:
            src = find_fallback(bn, args.fallback_root)
            if not src:
                raise SystemExit(
                    f"Annotated image {bn!r} not found in Full_dataset Anomaly "
                    f"folders nor under {args.fallback_root}")
            split = "train"          # not in the predefined split -> train
            fallback_used += 1
        copy_in(src, bn)

        w, h = bn_to_size[bn]
        img_id = next_img_id
        next_img_id += 1
        images_by_split[split].append(
            {"id": img_id, "width": w, "height": h, "file_name": bn})
        for ann in bn_to_anns[bn]:
            anns_by_split[split].append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"] + 1,   # 0 -> 1
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            })
            next_ann_id += 1

    # --- Negatives: normal images (zero boxes) ------------------------------
    for bn, (split, src) in sorted(normal_loc.items()):
        if bn in bn_to_anns:
            continue                 # safety: never treat an annotated img as neg
        copy_in(src, bn)
        with Image.open(src) as im:
            w, h = im.size
        images_by_split[split].append(
            {"id": next_img_id, "width": w, "height": h, "file_name": bn})
        next_img_id += 1

    print(f"Copied {copied} new image files into {img_dir} "
          f"({fallback_used} annotated images sourced from the fallback root)")

    # --- Write per-split COCO JSONs -----------------------------------------
    categories = [{"id": c["id"] + 1, "name": c["name"], "supercategory": ""}
                  for c in raw["categories"]]
    info = raw.get("info", {})
    for s in SPLITS:
        imgs = images_by_split[s]
        anns = anns_by_split[s]
        n_pos = sum(1 for im in imgs if any(True for _ in bn_to_anns.get(
            im["file_name"], [])))
        path = os.path.join(ann_dir, f"instances_{s}.json")
        with open(path, "w") as f:
            json.dump({"info": info, "images": imgs,
                       "annotations": anns, "categories": categories}, f)
        print(f"  {s:5s}: {len(imgs):4d} images "
              f"({n_pos} anomaly / {len(imgs) - n_pos} normal), "
              f"{len(anns):4d} boxes -> {path}")

    total_imgs = sum(len(images_by_split[s]) for s in SPLITS)
    total_anns = sum(len(anns_by_split[s]) for s in SPLITS)
    print(f"Done. {total_imgs} images / {total_anns} boxes across 3 splits.")


if __name__ == "__main__":
    main()
