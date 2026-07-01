"""
MS COCO 2017 dataset class for EfficientDet.

Returns per-sample:
    image    : FloatTensor[3, H, W]
    gt_boxes : FloatTensor[M, 4]  (cx, cy, w, h) absolute pixels
    gt_labels: LongTensor[M]      0-indexed class ids
"""
import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from efficientdet.utils.box_ops import xyxy_to_cxcywh


class CocoDataset(Dataset):
    def __init__(self, root: str, ann_file: str, transforms=None,
                 split=None, test_fraction: float = 0.05, seed: int = 42,
                 keep_empty: bool = False):
        """
        root     : directory containing the JPEG images
                   (e.g.  coco2017/train2017)
        ann_file : path to the COCO JSON annotation file
                   (e.g.  coco2017/annotations/instances_train2017.json)
        transforms : optional Compose from dataset.transforms
        keep_empty : if False (default) images with no annotations are dropped;
                   if True they are kept and yielded as background/negative
                   samples (zero boxes). Useful when the dataset deliberately
                   includes "normal" images to suppress false positives.
        split    : None  -> use every annotated image (e.g. the val2017 set)
                   "train" -> all annotated images EXCEPT the held-out test slice
                   "test"  -> only the held-out test slice
                   The held-out slice is carved deterministically from the
                   annotated images so that "train" and "test" are disjoint as
                   long as test_fraction / seed match across callers.
        test_fraction : fraction of annotated images reserved for the test split
        seed     : RNG seed controlling which images land in the test slice
        """
        self.root = root
        self.transforms = transforms
        self.coco = COCO(ann_file)

        # Keep only images that have at least one annotation (sorted for a
        # deterministic ordering independent of dict insertion order). When
        # keep_empty is set, annotation-free images are retained as negatives.
        if keep_empty:
            all_ids = sorted(self.coco.imgs.keys())
        else:
            all_ids = sorted(
                i for i in self.coco.imgs.keys()
                if len(self.coco.getAnnIds(imgIds=i)) > 0
            )

        if split in ("train", "test"):
            shuffled = all_ids[:]
            random.Random(seed).shuffle(shuffled)
            n_test = int(len(shuffled) * test_fraction)
            test_ids = set(shuffled[:n_test])
            if split == "test":
                self.ids = [i for i in all_ids if i in test_ids]
            else:  # "train"
                self.ids = [i for i in all_ids if i not in test_ids]
        elif split is None:
            self.ids = all_ids
        else:
            raise ValueError(f"split must be None, 'train', or 'test', got {split!r}")

        # Build a contiguous 0-indexed label map from COCO category ids
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid: idx for idx, cid in enumerate(cat_ids)}
        self.label_to_cat_id = {idx: cid for cid, idx in self.cat_id_to_label.items()}
        self.num_classes = len(cat_ids)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        path = os.path.join(self.root, img_info["file_name"])
        image = Image.open(path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes  = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # COCO stores (x1, y1, w, h)
            if w <= 0 or h <= 0:
                continue
            # Convert to (x1, y1, x2, y2) — used by transforms
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])

        boxes  = torch.tensor(boxes,  dtype=torch.float32)   # (M, 4) xyxy
        labels = torch.tensor(labels, dtype=torch.int64)      # (M,)

        if boxes.numel() == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        if self.transforms is not None:
            image, boxes, labels = self.transforms(image, boxes, labels)

        # Convert boxes from (x1, y1, x2, y2) → (cx, cy, w, h) for the model
        if boxes.numel() > 0:
            boxes = xyxy_to_cxcywh(boxes)

        return image, boxes, labels, img_id

    def get_num_classes(self) -> int:
        return self.num_classes
