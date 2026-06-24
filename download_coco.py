"""
Download and extract MS COCO 2017 into ./coco/

Run once before training:
    python download_coco.py

Sizes (approximate):
    train2017 images  : 18 GB
    val2017   images  :  1 GB
    test2017  images  :  6 GB
    annotations       :  0.2 GB

Pass --no-test to skip the test set (saves 6 GB).
"""
import argparse
import os
import urllib.request
import zipfile


URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images":   "http://images.cocodataset.org/zips/val2017.zip",
    "test_images":  "http://images.cocodataset.org/zips/test2017.zip",
    "annotations":  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _progress(block_count, block_size, total_size):
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1e6
        print(f"\r  {mb:.0f} MB  ({pct:.1f}%)", end="", flush=True)


def download_and_extract(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.join(dest_dir, url.split("/")[-1])

    if not os.path.exists(filename):
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, filename, reporthook=_progress)
        print()
    else:
        print(f"Already downloaded: {filename}")

    print(f"  Extracting {filename} ...")
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(filename)
    print(f"  Done.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dest",    default="coco", help="root directory for the dataset")
    p.add_argument("--no-test", action="store_true", help="skip test2017 images (~6 GB)")
    args = p.parse_args()

    root = args.dest
    imgs_dir = os.path.join(root, "images")

    download_and_extract(URLS["annotations"], root)
    download_and_extract(URLS["train_images"], imgs_dir)
    download_and_extract(URLS["val_images"],   imgs_dir)
    if not args.no_test:
        download_and_extract(URLS["test_images"], imgs_dir)

    print("Dataset ready. Expected layout:")
    print(f"  {root}/images/train2017/   (118 287 images)")
    print(f"  {root}/images/val2017/     (  5 000 images)")
    if not args.no_test:
        print(f"  {root}/images/test2017/    ( 40 670 images)")
    print(f"  {root}/annotations/instances_train2017.json")
    print(f"  {root}/annotations/instances_val2017.json")


if __name__ == "__main__":
    main()
