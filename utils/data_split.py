"""
Dataset Splitter Module
Splits image/label datasets into train, val, and test subsets.
"""

import os
import random
import shutil
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """
    Configuration dataclass for dataset split ratios and reproducibility.

    Attributes:
        train_ratio (float): Fraction of data for training (e.g., 0.7).
        val_ratio (float): Fraction of data for validation (e.g., 0.2).
        test_ratio (float): Fraction of data for testing (e.g., 0.1).
        seed (int): Random seed for reproducible shuffling.
    """

    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int = 42

    def __post_init__(self) -> None:
        """Validates that split ratios sum to approximately 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"[ERROR] Split ratios must sum to 1.0, got {total:.2f}"
            )


class DatasetSplitter:
    """
    Splits a dataset of images and labels into train, val, and test subsets.

    Expects the source directory to contain:
        source_dir/
            images/   ← image files (.jpg, .jpeg, .png, .bmp)
            labels/   ← corresponding .txt label files

    Attributes:
        source_dir (str): Root directory containing 'images' and 'labels' folders.
        output_dir (str): Root directory where split subsets will be written.
        config (SplitConfig): Ratio and seed configuration for the split.
    """

    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        config: SplitConfig,
    ) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.config = config

        self._images_dir = os.path.join(source_dir, "images")
        self._labels_dir = os.path.join(source_dir, "labels")

    def _validate_source(self) -> None:
        """Raises FileNotFoundError if source images or labels folders are missing."""
        for path in (self._images_dir, self._labels_dir):
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"[ERROR] Required directory not found: {path}"
                )

    def _collect_images(self) -> list[str]:
        """
        Returns a shuffled list of image filenames from the images directory.

        Returns:
            list[str]: Shuffled image filenames.
        """
        random.seed(self.config.seed)
        image_files = [
            f for f in os.listdir(self._images_dir)
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ]
        if not image_files:
            raise FileNotFoundError(
                f"[ERROR] No supported images found in: {self._images_dir}"
            )
        random.shuffle(image_files)
        return image_files

    def _compute_splits(
        self, image_files: list[str]
    ) -> dict[str, list[str]]:
        """
        Divides image filenames into train, val, and test groups.

        Args:
            image_files (list[str]): Full shuffled list of image filenames.

        Returns:
            dict[str, list[str]]: Mapping of split name to its image filenames.
        """
        total = len(image_files)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)

        return {
            "train": image_files[:train_end],
            "val": image_files[train_end:val_end],
            "test": image_files[val_end:],
        }

    def _create_output_dirs(self, splits: dict[str, list[str]]) -> None:
        """
        Creates output image and label folders for each split.

        Args:
            splits (dict[str, list[str]]): Split name to filenames mapping.
        """
        for split in splits:
            for sub in ("images", "labels"):
                os.makedirs(
                    os.path.join(self.output_dir, split, sub), exist_ok=True
                )

    def _copy_files(self, splits: dict[str, list[str]]) -> None:
        """
        Copies image and label files into their respective split folders.

        Args:
            splits (dict[str, list[str]]): Split name to filenames mapping.
        """
        for split, files in splits.items():
            for img_file in files:
                self._copy_image(split, img_file)
                self._copy_label(split, img_file)

    def _copy_image(self, split: str, img_file: str) -> None:
        """
        Copies a single image file to its split output directory.

        Args:
            split (str): Name of the split ('train', 'val', or 'test').
            img_file (str): Image filename to copy.
        """
        src = os.path.join(self._images_dir, img_file)
        dst = os.path.join(self.output_dir, split, "images", img_file)
        try:
            shutil.copy2(src, dst)
        except OSError as e:
            print(f"[WARNING] Could not copy image '{img_file}': {e}")

    def _copy_label(self, split: str, img_file: str) -> None:
        """
        Copies the corresponding label .txt file for a given image.

        Args:
            split (str): Name of the split ('train', 'val', or 'test').
            img_file (str): Image filename whose label should be copied.
        """
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src = os.path.join(self._labels_dir, label_file)
        dst = os.path.join(self.output_dir, split, "labels", label_file)

        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                print(f"[WARNING] Could not copy label '{label_file}': {e}")
        else:
            print(f"[WARNING] No label found for image: {img_file}")

    def _print_summary(self, splits: dict[str, list[str]]) -> None:
        """
        Prints a summary of how many files landed in each split.

        Args:
            splits (dict[str, list[str]]): Split name to filenames mapping.
        """
        total = sum(len(v) for v in splits.values())
        print(f"[INFO] Split complete. Total: {total} images")
        for split, files in splits.items():
            print(f"  {split.capitalize():<6}: {len(files)}")

    def run(self) -> None:
        """
        Orchestrates the full dataset splitting pipeline:
        validate → collect → split → create dirs → copy → summarize.
        """
        try:
            self._validate_source()
            image_files = self._collect_images()
            splits = self._compute_splits(image_files)
            self._create_output_dirs(splits)
            self._copy_files(splits)
            self._print_summary(splits)
        except FileNotFoundError as e:
            print(f"[ERROR] Source validation failed: {e}")
        except ValueError as e:
            print(f"[ERROR] Configuration error: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error during split: {e}")
            raise


def main() -> None:
    """Entry point for the dataset splitter application."""
    config = SplitConfig(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )

    splitter = DatasetSplitter(
        source_dir="data/raw",
        output_dir="data/split",
        config=config,
    )
    splitter.run()


if __name__ == "__main__":
    main()