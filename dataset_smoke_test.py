"""
Quick smoke test for the CIFAR and Food-101 dataset loaders.

Each test pulls a tiny batch from the train/val/test iterators and reports
basic statistics so we can spot obvious hangs or shape mismatches early.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np

import Dataset as cifar_dataset
import Dataset_foods as food_dataset


def describe_split(iterator: Iterable[Tuple[np.ndarray, np.ndarray]], label: str) -> None:
    start = time.perf_counter()
    batch_images, batch_labels = next(iterator)
    elapsed = time.perf_counter() - start
    print(
        f"  {label}: images {batch_images.shape}, labels {batch_labels.shape}, "
        f"min {batch_images.min():.3f}, max {batch_images.max():.3f}, "
        f"time {elapsed:.3f}s",
    )


def inspect_bundle(name: str, bundle: Dict[str, Any], *, take_train: bool = True) -> None:
    print(f"\n{name}")
    print(
        f"  sizes train={bundle['train_size']} val={bundle['val_size']} "
        f"test={bundle['test_size']} image_shape={bundle['image_shape']}"
    )
    train_iter = bundle["train_iter_fn"]()
    if take_train:
        describe_split(train_iter, "train batch")
    val_iter = bundle["val_iter_fn"]()
    describe_split(val_iter, "val batch")
    test_iter = bundle["test_iter_fn"]()
    describe_split(test_iter, "test batch")


def main() -> None:
    start = time.perf_counter()
    cifar_bundle = cifar_dataset.get_dataset(batch_size=8, seed=0)
    inspect_bundle("CIFAR-10", cifar_bundle)

    food_bundle = food_dataset.get_dataset(
        batch_size=2,
        seed=0,
        val_samples=256,
        target_height=128,
        target_width=128,
    )
    inspect_bundle("Food-101 (streamed)", food_bundle, take_train=False)
    elapsed = time.perf_counter() - start
    print(f"\nTotal elapsed {elapsed:.2f}s")


if __name__ == "__main__":
    main()
