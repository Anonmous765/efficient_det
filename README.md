# EfficientDet Anomaly Detection

A from-scratch PyTorch implementation of [EfficientDet](https://arxiv.org/abs/1911.09070) (EfficientNet backbone + BiFPN + class/box heads), with two training pipelines:

1. **MS COCO 2017** — general object detection, used to validate the model implementation.
2. **Custom anomaly dataset** — 3D-print defect detection, treating "Normal" prints as negative (background) samples so the model learns not to fire on clean parts.

## Architecture

```
efficientdet/
├── backbone.py       EfficientNet backbone (via timm), feature maps P3-P5
├── bifpn/             Bi-directional Feature Pyramid Network (BiFPN)
│   ├── layer.py         Single BiFPN layer (top-down + bottom-up fusion, stacked num_bifpn_layers times)
│   └── fusion.py        Fast normalized feature fusion (weighted sum, ReLU'd learnable weights)
├── heads.py           Classification and box regression heads (shared weights across pyramid levels)
├── config.py          Compound-scaled config (phi 0-7 controls width/depth/resolution)
├── model.py           EfficientDet: wires backbone -> BiFPN -> heads -> anchors
└── utils/
    ├── anchors.py      Anchor box generation (multi-scale, multi-aspect-ratio, per pyramid level)
    ├── matcher.py       Anchor <-> ground-truth matching (IoU-based, foreground/background/ignore)
    ├── loss.py           Focal loss (classification) + L1 loss (box regression)
    ├── nms.py            Post-processing: score threshold + NMS
    ├── box_ops.py        Box format conversions (xyxy <-> cxcywh)
    └── conv.py           Conv building blocks (e.g. depthwise separable)

dataset/
├── coco.py            COCO-format dataset loader, shared by both pipelines
├── transforms.py       Resize, RandomHorizontalFlip, ColorJitter, ToTensor
└── collate.py           Batch collation for variable-length annotations

train.py                 Training loop (both pipelines)
evaluate.py               COCO mAP evaluation
visualize.py               Draws predicted vs. ground-truth boxes to PNGs
download_coco.py           Downloads/extracts MS COCO 2017
prepare_anomaly_data.py    Converts the Label Studio export + Normal/Anomaly folders into COCO format
```

The backbone scale is controlled by `--phi` (0–7), following the EfficientDet compound scaling rule: higher `phi` means a larger EfficientNet backbone, more BiFPN layers, more head layers, and a larger input resolution.

| `phi` | backbone | input resolution | BiFPN layers | head layers | BiFPN channels |
|-------|----------|-------------------|---------------|-------------|-----------------|
| 0 | efficientnet_b0 | 512 | 3 | 3 | 64 |
| 1 | efficientnet_b1 | 640 | 4 | 3 | 88 |
| 2 | efficientnet_b2 | 768 | 5 | 3 | 112 |
| 3 | efficientnet_b3 | 896 | 6 | 4 | 160 |
| 4 | efficientnet_b4 | 1024 | 7 | 4 | 224 |
| 5 | efficientnet_b5 | 1152 | 8 | 4 | 296 |
| 6 | efficientnet_b6 | 1280 | 9 | 5 | 384 |
| 7 | efficientnet_b6 | 1408 | 10 | 5 | 384 |

(`out_channels`, `num_bifpn_layers`, `num_head_layers`, and `input_resolution` are all derived formulaically from `phi` in `efficientdet/config.py`; the table above shows the resulting values.)

## Setup

```bash
pip install torch torchvision timm pycocotools pillow matplotlib
```

- A CUDA GPU is recommended for training; `train.py` auto-detects `cuda` and falls back to CPU otherwise.
- No `requirements.txt`/`pyproject.toml` is checked in — the packages above are the full runtime dependency set.
- Training uses bf16 autocast by default, so a CUDA device with bf16 support (Ampere or newer) gets the most benefit; `--no-amp` disables it for older GPUs.

## Pipeline 1: MS COCO 2017

**1. Download the dataset**

```bash
python download_coco.py --dest coco2017        # ~25 GB (--no-test skips test2017, saves 6 GB)
```

| flag | default | description |
|------|---------|-------------|
| `--dest` | `coco` | root directory the dataset is extracted into |
| `--no-test` | off | skip `test2017` images (~6 GB) since they have no annotations and aren't needed for training/eval |

**2. Train**

```bash
python train.py \
    --train-images coco2017/images/train2017 \
    --train-ann    coco2017/annotations/instances_train2017.json \
    --val-images   coco2017/images/val2017 \
    --val-ann      coco2017/annotations/instances_val2017.json \
    --phi 0 --epochs 300 --batch-size 8 --lr 1e-4 --workers 4 \
    --checkpoint-dir checkpoints
```

A held-out test slice (`--test-fraction`, default 5%) is carved out of `train2017` and never trained on, so it can be used for a final evaluation independent of `val2017`.

| flag | default | description |
|------|---------|-------------|
| `--train-images` | `coco2017/train2017` | training image directory |
| `--train-ann` | `coco2017/annotations/instances_train2017.json` | training COCO annotation file |
| `--val-images` | `coco2017/val2017` | validation image directory |
| `--val-ann` | `coco2017/annotations/instances_val2017.json` | validation COCO annotation file |
| `--test-fraction` | `0.05` | fraction of `train2017` carved out as a held-out test split (never trained on) |
| `--split-seed` | `42` | RNG seed for the train/test carve-out; must match `evaluate.py`/`visualize.py` to reproduce the same split |
| `--keep-empty` | off | keep images with zero annotations as negative/background samples (required for the anomaly pipeline) |
| `--phi` | `0` | EfficientDet compound scaling coefficient, 0–7 (see table above) |
| `--epochs` | `50` | number of training epochs |
| `--batch-size` | `16` | batch size **per GPU** (under `torchrun` the effective batch is this × number of processes) |
| `--lr` | `1e-4` | learning rate |
| `--weight-decay` | `1e-4` | weight decay |
| `--workers` | `4` | `DataLoader` worker processes **per GPU process** |
| `--checkpoint-dir` | `checkpoints` | directory for checkpoints, loss curve, and loss history |
| `--resume` | none | path to a checkpoint to resume training **the same run** from (restores model, optimizer, and loss history) |
| `--init-from` | none | path to a checkpoint to **initialize weights from** for a new run on a different dataset (e.g. COCO → anomaly transfer learning). Loads only shape-matching tensors — the classification head is skipped when `num_classes` differs and stays randomly initialized. Optimizer state and epoch count are not restored. Mutually exclusive with `--resume` |
| `--no-amp` | off | disable bf16 mixed-precision training |
| `--compile` | off | wrap the model with `torch.compile` |
| `--patience` | `0` (disabled) | stop early after `N` epochs without validation-loss improvement |
| `--freeze-backbone` | off | freeze the EfficientNet backbone (weights + BatchNorm running stats); only the BiFPN and heads train. Typically paired with `--init-from` when fine-tuning on a small dataset |
| `--find-unused-parameters` | off | multi-GPU only; see the DDP section below |
| `--dist-timeout` | `120` | multi-GPU only; seconds a collective may block before failing (NCCL's own default is 600) |

**Multi-GPU training (DistributedDataParallel)**

`train.py` runs on multiple GPUs via `torchrun`, one process per GPU. No code changes or extra flags are needed — launching under `torchrun` is what switches it on, and a plain `python train.py` still runs single-process exactly as before:

```bash
torchrun --nproc_per_node=4 train.py \
    --train-images data/anomaly/images \
    --train-ann    data/anomaly/annotations/instances_train.json \
    --val-images   data/anomaly/images \
    --val-ann      data/anomaly/annotations/instances_val.json \
    --phi 0 --test-fraction 0 --keep-empty \
    --batch-size 8 --lr 2e-4
```

Set `--nproc_per_node` to the number of GPUs you want to use. What changes under DDP:

- **`--batch-size` and `--workers` are per GPU.** The example above trains on an effective batch of 8 × 4 = 32, so the learning rate is scaled up to compensate.
- **A `DistributedSampler` shards each epoch** so every rank sees a disjoint subset of images; the sampler is re-seeded each epoch (`set_epoch`) so the shuffling actually changes between epochs.
- **Reported losses are averaged across ranks.** Each rank only sees its own shard, so `train_loss`/`val_loss` are all-reduced before being printed. This also keeps best-checkpoint selection and `--patience` early stopping identical on every rank — if the ranks disagreed, one could exit the epoch loop while the others hung waiting on it.
- **Only rank 0 prints, writes checkpoints, and plots the loss curve**, so the checkpoint directory isn't written by four processes at once.
- **Checkpoints stay wrapper-free.** The bare model is saved, not the DDP wrapper, so `state_dict` keys have no `module.` prefix and `evaluate.py` / `visualize.py` load them unchanged.

If training crashes with *"Expected to have finished reduction in the prior iteration"*, add `--find-unused-parameters`. It costs some speed, which is why it's off by default.

A benign `UserWarning: Grad strides do not match bucket view strides` may appear — it comes from combining `channels_last` memory format with DDP's gradient buckets and does not affect correctness.

**Troubleshooting: NCCL hangs at startup**

On multi-GPU workstations without NVLink (e.g. several RTX A5000s on plain PCIe), NCCL often cannot establish peer-to-peer between GPUs, usually because IOMMU/ACS blocks it. Training then hangs at the `DistributedDataParallel(...)` line — the first NCCL collective — and eventually dies with:

```
RuntimeError: DDP expects same model across all ranks, but Rank 3 has 419 params,
              while rank 0 has inconsistent 0 params.
WorkNCCL(... OpType=ALLGATHER ...) ran for 600030 milliseconds before timing out
```

The "inconsistent 0 params" is a red herring — the models are fine, the ALLGATHER simply never returned. The fix is to route collectives through host memory:

```bash
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 train.py ...
```

Beware that a hung NCCL collective **does not look idle**: its spin-wait kernels hold every GPU at 100% utilization with high power draw and temperature, which is easily mistaken for real training. Check `checkpoints/loss_history.json` for actual epoch progress rather than trusting `nvidia-smi`.

To confirm the diagnosis in seconds rather than minutes, test NCCL on its own:

```bash
cat > /tmp/nccl_test.py <<'EOF'
import os, torch, torch.distributed as dist
from datetime import timedelta
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
t = torch.ones(1, device=f"cuda:{lr}")
dist.all_reduce(t)
print(f"rank {dist.get_rank()}: all_reduce OK -> {t.item()}", flush=True)
dist.destroy_process_group()
EOF

torchrun --nproc_per_node=4 /tmp/nccl_test.py                      # hangs if P2P is broken
NCCL_P2P_DISABLE=1 torchrun --nproc_per_node=4 /tmp/nccl_test.py   # should print 4.0 per rank
```

`nvidia-smi topo -m` shows the interconnect matrix if you want to see how the GPUs are wired.

**3. Evaluate (COCO mAP)**

```bash
python evaluate.py \
    --images coco2017/images/train2017 \
    --ann    coco2017/annotations/instances_train2017.json \
    --split test --checkpoint checkpoints/best.pth --phi 0
```

`--test-fraction` / `--split-seed` must match the values used in `train.py` so the held-out split lines up.

| flag | default | description |
|------|---------|-------------|
| `--images` | `coco2017/train2017` | image directory to evaluate on |
| `--ann` | `coco2017/annotations/instances_train2017.json` | COCO annotation file |
| `--split` | `test` | which slice to evaluate: `test`, `train`, or `all` |
| `--test-fraction` | `0.05` | must match the value used in `train.py` |
| `--split-seed` | `42` | must match the value used in `train.py` |
| `--checkpoint` | `checkpoints/best.pth` | checkpoint to evaluate |
| `--phi` | `0` | must match the checkpoint's `phi` |
| `--batch-size` | `8` | evaluation batch size |
| `--workers` | `4` | `DataLoader` worker processes |
| `--score-thresh` | `0.05` | minimum confidence score kept before NMS |
| `--iou-thresh` | `0.5` | NMS IoU threshold |
| `--keep-empty` | off | include zero-annotation images (needed so false positives on Normal/negative images count in the anomaly pipeline) |
| `--decision-thresh` | best F1 | score threshold for the image-level confusion matrix; defaults to whichever swept value maximizes F1 |
| `--no-image-level` | off | skip the image-level confusion matrix and report only COCO mAP |

`evaluate.py` reports two complementary things:

**Box-level (COCO mAP)** — how well boxes are localized, the standard detection metric.

**Image-level confusion matrix** — treats each image as one binary decision: *does this image contain an anomaly?* An image is flagged anomalous when its highest-scoring detection clears the decision threshold, and is truly anomalous when it carries at least one ground-truth box. This is the metric that matches how the model gets used in practice ("stop the print / don't stop the print"), and it only means anything when the negatives are present — so pass `--keep-empty`.

It prints a sweep over decision thresholds, then a matrix at the best-F1 threshold (or `--decision-thresh`):

```
Confusion matrix @ score >= 0.45  (best-F1 threshold)
                     predicted
                 anomaly   normal
  actual anomaly      52       11
  actual normal        9      110

  accuracy    0.8901
  precision   0.8525   (of images flagged, this fraction really was defective)
  recall      0.8254   (of defective prints, this fraction was caught)
  specificity 0.9244   (of good prints, this fraction was left alone)
  F1          0.8387
```

Read **recall** and **specificity** rather than accuracy: the anomaly test split is 63 anomaly / 119 normal, so a model that detects nothing at all still scores 65% accuracy. Precision/recall trade against each other as the threshold moves — lower it to miss fewer defects at the cost of more false alarms.

Runs pycocotools' `COCOeval` and prints the standard COCO mAP summary (AP@[.5:.95], AP50, AP75, AP by size, AR).

**4. Visualize predictions**

```bash
python visualize.py --checkpoint checkpoints/best.pth --split test --num-images 12
```

Saves PNGs to `predictions/` with predicted boxes in red and ground-truth boxes in green.

| flag | default | description |
|------|---------|-------------|
| `--checkpoint` | `checkpoints/best.pth` | checkpoint to visualize |
| `--images` | `coco2017/train2017` | image directory |
| `--ann` | `coco2017/annotations/instances_train2017.json` | COCO annotation file |
| `--split` | `test` | which slice to sample from: `test`, `train`, or `all` |
| `--test-fraction` | `0.05` | must match the value used in `train.py` |
| `--split-seed` | `42` | must match the value used in `train.py` |
| `--phi` | `0` | must match the checkpoint's `phi` |
| `--num-images` | `12` | number of images to sample and render |
| `--score-thresh` | `0.3` | minimum confidence score for a prediction to be drawn |
| `--iou-thresh` | `0.5` | NMS IoU threshold |
| `--no-gt` | off | don't draw ground-truth boxes, predictions only |
| `--out-dir` | `predictions` | output directory for the rendered PNGs |

## Pipeline 2: Custom anomaly dataset

The raw data consists of a Label Studio COCO export (`result.json`, the bounding-box source of truth) plus a `Full_dataset/training_data` folder with a predefined `<split>/Anomaly` and `<split>/Normal` layout.

**1. Prepare the dataset**

```bash
python prepare_anomaly_data.py \
    --result-json result.json \
    --full-dataset ~/Desktop/images/Full_dataset/training_data \
    --out-dir data/anomaly
```

This flattens images into `data/anomaly/images/`, adds `Normal` images as zero-annotation negatives, remaps category ids to 1-indexed COCO, and writes `data/anomaly/annotations/instances_{train,val,test}.json`.

| flag | default | description |
|------|---------|-------------|
| `--result-json` | `result.json` | Label Studio COCO export, source of truth for bounding boxes |
| `--full-dataset` | *(required)* | path to `Full_dataset/training_data`, containing `<split>/Anomaly` and `<split>/Normal` |
| `--fallback-root` | none | secondary root to search for images not found under `--full-dataset` |
| `--out-dir` | `data/anomaly` | output directory for flattened images + COCO annotation files |

**2. Train**

```bash
python train.py \
    --train-images data/anomaly/images \
    --train-ann    data/anomaly/annotations/instances_train.json \
    --val-images   data/anomaly/images \
    --val-ann      data/anomaly/annotations/instances_val.json \
    --phi 0 --test-fraction 0 --keep-empty
```

`--keep-empty` is required so that negative (Normal) images are used as background samples during training. `--test-fraction 0` disables the train/test carve-out since the anomaly dataset already ships with a predefined split. All other `train.py` flags (see the table in Pipeline 1) apply the same way here.

**2b. (Recommended) Initialize from a COCO-pretrained checkpoint**

The anomaly dataset is small (~2.2k images, 1 class), so training it from scratch wastes everything the model could otherwise learn about general image features from COCO's 118k images. Use `--init-from` to carry over the COCO-pretrained backbone + BiFPN weights and fine-tune on top of them, instead of `--resume` (which is for continuing an interrupted run on the *same* dataset, not for transferring to a new one):

```bash
python train.py \
    --train-images data/anomaly/images \
    --train-ann    data/anomaly/annotations/instances_train.json \
    --val-images   data/anomaly/images \
    --val-ann      data/anomaly/annotations/instances_val.json \
    --phi 0 --test-fraction 0 --keep-empty \
    --init-from checkpoints_coco/best.pth \
    --checkpoint-dir checkpoints_anomaly \
    --lr 1e-5 --epochs 100
```

Why this works: COCO has 80 classes and the anomaly dataset has 1, so the two models' classification heads are different shapes (`final_conv` in `ClassificationHead`, `efficientdet/heads.py`). `--init-from` loads every tensor whose shape matches the freshly built model — that's the entire EfficientNet backbone and BiFPN, plus the box head and all of the classification head except its last layer — and leaves the one mismatched layer (the class-count-dependent `final_conv`) randomly initialized. It prints how many tensors were transferred vs. skipped so you can confirm the load did what you expect. A lower `--lr` than the from-scratch default is typical for fine-tuning a mostly-pretrained model.

Add `--freeze-backbone` if the anomaly dataset is small enough that fine-tuning the full EfficientNet backbone risks overfitting or destroying its pretrained features — this locks the backbone's weights and BatchNorm running stats, training only the BiFPN and heads on top of frozen COCO-pretrained features:

```bash
python train.py \
    --train-images data/anomaly/images \
    --train-ann    data/anomaly/annotations/instances_train.json \
    --val-images   data/anomaly/images \
    --val-ann      data/anomaly/annotations/instances_val.json \
    --phi 0 --test-fraction 0 --keep-empty \
    --init-from checkpoints_coco/best.pth --freeze-backbone \
    --checkpoint-dir checkpoints_anomaly \
    --lr 1e-4 --epochs 100
```

**3. Evaluate / visualize**

Same as the COCO pipeline — point `evaluate.py` / `visualize.py` at `data/anomaly/images` and `data/anomaly/annotations/instances_test.json` (use `--keep-empty` so false positives on Normal images are counted).

## Notes

- Checkpoints (`last.pth` / `best.pth`), a loss curve, and loss history are written to `--checkpoint-dir` (default `checkpoints/`) after every epoch. `best.pth` tracks the lowest validation loss seen so far; `last.pth` is always the most recent epoch, so it's the one `--resume` should generally point at.
- Training uses bf16 mixed precision on CUDA by default (`--no-amp` to disable), TF32 matmuls, and `channels_last` memory format.
- `--resume <checkpoint>` continues training from a saved checkpoint, including its loss history. `--init-from <checkpoint>` instead starts a *new* run initialized from another checkpoint's weights (shape-matching tensors only) — use this for COCO → anomaly transfer learning, not `--resume`, since the two datasets have different `num_classes`.
- `--patience N` enables early stopping after `N` epochs without validation-loss improvement.
- Loss is the sum of focal loss (classification, handles foreground/background class imbalance) and L1 loss (box regression), computed only over anchors matched to a ground-truth box by `efficientdet/utils/matcher.py`.
- Datasets, checkpoints, and predictions are gitignored — this repo tracks code only.
