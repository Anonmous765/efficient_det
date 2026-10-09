"""
Train anomalib's EfficientAd on the 3D-print anomaly dataset, then test it.

Expects the folder layout written by ``efficientAD.prepare_folders``. Training
uses only the Normal images of the train split. The val split (normal +
defect) is used during fit to pick the anomaly-score threshold and the
normalisation stats; the held-out test split is scored once at the end.

EfficientAd also trains its student against ImageNette as a penalty set; it is
downloaded (~1.5 GB) into --imagenet-dir on the first run.

Usage (from the repo root):
    python -m efficientAD.train --epochs 20

Results (checkpoint, metrics, heat-map images) go to --results-dir.
"""
import argparse

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd


def make_datamodule(root, eval_split, workers):
    """Folder datamodule: train on train/good, validate/test on <eval_split>/{good,defect}."""
    return Folder(
        name="3d_print",
        root=root,
        normal_dir="train/good",
        abnormal_dir=f"{eval_split}/defect",
        normal_test_dir=f"{eval_split}/good",
        mask_dir=f"{eval_split}/mask",
        train_batch_size=1,  # EfficientAd requires batch size 1
        eval_batch_size=16,
        num_workers=workers,
        val_split_mode="same_as_test",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/anomaly_folder")
    parser.add_argument("--imagenet-dir", default="datasets/imagenette")
    parser.add_argument("--model-size", choices=("small", "medium"), default="small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--results-dir", default="efficientAD/results")
    args = parser.parse_args()

    model = EfficientAd(imagenet_dir=args.imagenet_dir, model_size=args.model_size)
    engine = Engine(max_epochs=args.epochs, default_root_dir=args.results_dir)

    # The val split sets the threshold; testing on a separate split keeps it honest.
    engine.fit(model=model, datamodule=make_datamodule(args.data_root, "val", args.workers))
    engine.test(model=model, datamodule=make_datamodule(args.data_root, "test", args.workers))


if __name__ == "__main__":
    main()
