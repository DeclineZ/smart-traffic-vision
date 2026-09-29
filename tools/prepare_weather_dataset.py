"""
Automated Weather Dataset Downloader and Preprocessor for YOLOv8-cls.
Downloads the Multi-Class Weather Dataset and prepares train/val splits
categorized into 'rainy' and 'clear' (non-rainy) classes.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DEFAULT_DATASET_URL = (
    "https://github.com/PerceptiLabs/Weather-Analysis/archive/refs/heads/main.zip"
)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress reporting."""
    print(f"[+] Downloading dataset from: {url}")
    print(f"    Target path: {output_path}")

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (SmartTraffic/1.0)")]
    urllib.request.install_opener(opener)

    def _progress_hook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r    Progress: {percent:5.1f}% ({mb_downloaded:6.1f}MB / {mb_total:6.1f}MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=_progress_hook)
    print("\n[+] Download completed successfully.")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip archive safely."""
    print(f"[+] Extracting archive: {zip_path} -> {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("[+] Extraction completed.")


def is_valid_image(file_path: Path) -> bool:
    """Verify that an image file is valid and not empty."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if file_path.suffix.lower() not in valid_extensions:
        return False
    if file_path.stat().st_size < 1024:
        return False
    if HAS_PIL:
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    return True


def collect_and_organize(
    raw_root: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """
    Organizes images into YOLO classification structure:
    output_dir/
      train/
        clear/
        rainy/
      val/
        clear/
        rainy/
    """
    random.seed(seed)

    # Search for categories in extracted folder
    # Categories: 'rain' -> rainy, 'cloud' + 'shine' + 'sunrise' -> clear
    print(f"[+] Scanning for weather images in: {raw_root}")

    rain_images: List[Path] = []
    clear_images: List[Path] = []

    for path in raw_root.rglob("*"):
        if not path.is_file() or not is_valid_image(path):
            continue

        parent_name = path.parent.name.lower()
        if parent_name == "rain":
            rain_images.append(path)
        elif parent_name in ("cloud", "shine", "sunrise"):
            clear_images.append(path)

    print(f"    Found raw rainy images: {len(rain_images)}")
    print(f"    Found raw clear/non-rainy images: {len(clear_images)}")

    if not rain_images or not clear_images:
        raise RuntimeError(
            f"Could not find sufficient images in {raw_root}. "
            f"Check if the extraction folder structure matches expected names (rain, cloud, shine)."
        )

    # Balance datasets slightly if clear is much larger than rain
    random.shuffle(rain_images)
    random.shuffle(clear_images)

    # Prepare output directories
    splits = ["train", "val"]
    classes = ["rainy", "clear"]
    for split in splits:
        for cls in classes:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    def split_and_copy(images: List[Path], cls_name: str) -> Tuple[int, int]:
        n_train = int(len(images) * train_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:]

        for img in train_imgs:
            dest = output_dir / "train" / cls_name / f"{cls_name}_{img.name}"
            shutil.copy2(img, dest)

        for img in val_imgs:
            dest = output_dir / "val" / cls_name / f"{cls_name}_{img.name}"
            shutil.copy2(img, dest)

        return len(train_imgs), len(val_imgs)

    rain_train, rain_val = split_and_copy(rain_images, "rainy")
    clear_train, clear_val = split_and_copy(clear_images, "clear")

    print("\n" + "=" * 50)
    print("      DATASET PREPARATION SUMMARY")
    print("=" * 50)
    print(f"Output Directory: {output_dir.resolve()}")
    print(f"{'Split':<10} | {'Rainy':<10} | {'Clear':<10} | {'Total':<10}")
    print("-" * 50)
    print(
        f"{'train':<10} | {rain_train:<10} | {clear_train:<10} | {rain_train + clear_train:<10}"
    )
    print(
        f"{'val':<10} | {rain_val:<10} | {clear_val:<10} | {rain_val + clear_val:<10}"
    )
    print("-" * 50)
    print(
        f"{'TOTAL':<10} | {rain_train + rain_val:<10} | {clear_train + clear_val:<10} | "
        f"{rain_train + rain_val + clear_train + clear_val:<10}"
    )
    print("=" * 50)
    print("\n[✔] Dataset is ready for YOLOv8-cls training!")


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize Weather Dataset for YOLO classification."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DATASET_URL,
        help="Direct URL to dataset zip archive.",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to an existing local zip archive (skips download if provided).",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to an existing unzipped raw dataset folder (skips download & extraction).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/weather_dataset",
        help="Path to store organized train/val YOLO classification dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/Val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--clean-temp",
        action="store_true",
        default=True,
        help="Clean temporary download and extract cache after organizing.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else project_root / args.output_dir
    )
    temp_dir = project_root / "temp" / "weather_download"

    raw_data_dir: Path | None = None

    if args.source_dir:
        raw_data_dir = Path(args.source_dir).resolve()
        if not raw_data_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {raw_data_dir}")
    else:
        temp_dir.mkdir(parents=True, exist_ok=True)
        zip_file = (
            Path(args.zip_path).resolve()
            if args.zip_path
            else temp_dir / "weather_dataset.zip"
        )

        if not args.zip_path and not zip_file.exists():
            download_file(args.url, zip_file)

        extracted_dir = temp_dir / "extracted"
        if not extracted_dir.exists() or not any(extracted_dir.iterdir()):
            extract_zip(zip_file, extracted_dir)

        raw_data_dir = extracted_dir

    collect_and_organize(
        raw_root=raw_data_dir,
        output_dir=output_path,
        train_ratio=args.train_ratio,
    )

    if args.clean_temp and temp_dir.exists() and not args.zip_path and not args.source_dir:
        print(f"[+] Cleaning temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
