# 3D-Print Anomaly Detection

Compares detection architectures on a custom 3D-print defect dataset. Each architecture lives in its own self-contained folder; the datasets and the scripts that build them are shared at the repo root.

## Layout

```
data/                 Prepared anomaly dataset (gitignored), built by data_prep/prepare_anomaly_data.py
coco2017/             MS COCO 2017 (gitignored), downloaded by data_prep/download_coco.py
result.json           Label Studio COCO export: the source of truth for the anomaly boxes
data_prep/            Dataset scripts shared by every architecture
├── download_coco.py
└── prepare_anomaly_data.py
efficientdet/         EfficientDet: model, dataset loader, train/evaluate/visualize, results
```

Each architecture folder is a Python package with its own README. Run everything **from the repo root** so the shared data paths resolve, invoking scripts as modules, e.g. `python -m efficientdet.train`.

| architecture | folder | README |
|--------------|--------|--------|
| EfficientDet (box-level detector) | `efficientdet/` | [efficientdet/README.md](efficientdet/README.md) |

## Shared data

**Anomaly dataset.** This flattens the images into `data/anomaly/images/`, adds `Normal` images as zero-annotation negatives, and writes `data/anomaly/annotations/instances_{train,val,test}.json`:

```bash
python data_prep/prepare_anomaly_data.py \
    --result-json result.json \
    --full-dataset ~/Desktop/images/Full_dataset/training_data \
    --out-dir data/anomaly
```

**MS COCO 2017** (optional, used for EfficientDet pretraining):

```bash
python data_prep/download_coco.py --dest coco2017 --no-test
```

See [efficientdet/README.md](efficientdet/README.md) for every flag of both scripts.

## Adding an architecture

Create a new top-level package (e.g. `efficientad/`) that holds its own model code, data loading, entry-point scripts, README, and outputs (checkpoints, predictions). Read data from `data/` rather than copying it. Packages should not import from each other, so each one can change without breaking the others.
