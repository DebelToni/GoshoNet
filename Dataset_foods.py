from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
from datasets import IterableDataset, load_dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_NAME = "ethz/food101"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TRAIN_SIZE = 75750
VAL_SIZE = 25250
NUM_CLASSES = 101
DEFAULT_VAL_SAMPLES = 5000
DEFAULT_SHUFFLE_BUFFER = 8192

_CLASS_NAMES = None


def _class_names():
    global _CLASS_NAMES
    if _CLASS_NAMES is None:
        meta = load_dataset(DATASET_NAME, split="train[:1]")
        _CLASS_NAMES = list(meta.features["label"].names)
    return _CLASS_NAMES


def _prepare_image(image: Image.Image, target_hw: Tuple[int, int]) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    h, w = target_hw
    if image.size != (w, h):
        image = image.resize((w, h), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def _build_stream(
    split: str,
    *,
    shuffle: bool,
    seed: int,
    take: Optional[int] = None,
    skip: Optional[int] = None,
    buffer_size: int = DEFAULT_SHUFFLE_BUFFER,
) -> IterableDataset:
    ds: IterableDataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    if take:
        ds = ds.take(take)
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed)
    return ds


def _iter_dataset(
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    repeat: bool,
    seed: int,
    target_hw: Tuple[int, int],
    take: Optional[int] = None,
    skip: Optional[int] = None,
    buffer_size: int = DEFAULT_SHUFFLE_BUFFER,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def generator():
        local_seed = seed
        while True:
            stream = _build_stream(
                split,
                shuffle=shuffle,
                seed=local_seed,
                take=take,
                skip=skip,
                buffer_size=buffer_size,
            )
            batch_images = []
            batch_labels = []

            for example in stream:
                processed = _prepare_image(example["image"], target_hw)
                batch_images.append(processed)
                batch_labels.append(int(example["label"]))

                if len(batch_images) == batch_size:
                    yield np.stack(batch_images, axis=0), np.asarray(batch_labels, dtype=np.int32)
                    batch_images.clear()
                    batch_labels.clear()

            if batch_images:
                batch_images.clear()
                batch_labels.clear()

            if not repeat:
                break

            if shuffle:
                local_seed += 1

    return generator()


def get_dataset(
    batch_size: int,
    *,
    seed: int = 0,
    val_samples: int = DEFAULT_VAL_SAMPLES,
    target_height: int = 256,
    target_width: int = 256,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER,
    **_,
) -> Dict[str, object]:
    if val_samples <= 0 or val_samples >= VAL_SIZE:
        raise ValueError(f"val_samples must be between 1 and {VAL_SIZE - 1}")

    target_hw = (target_height, target_width)

    train_iter_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _iter_dataset(
        TRAIN_SPLIT,
        batch_size=batch_size,
        shuffle=True,
        repeat=True,
        seed=seed,
        target_hw=target_hw,
        buffer_size=shuffle_buffer_size,
    )

    train_iter_no_shuffle_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _iter_dataset(
        TRAIN_SPLIT,
        batch_size=batch_size,
        shuffle=False,
        repeat=True,
        seed=seed,
        target_hw=target_hw,
        buffer_size=shuffle_buffer_size,
    )

    val_iter_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _iter_dataset(
        VAL_SPLIT,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        seed=seed,
        target_hw=target_hw,
        take=val_samples,
        buffer_size=shuffle_buffer_size,
    )

    test_iter_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _iter_dataset(
        VAL_SPLIT,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        seed=seed,
        target_hw=target_hw,
        skip=val_samples,
        buffer_size=shuffle_buffer_size,
    )

    bundle = {
        "train_iter_fn": train_iter_fn,
        "train_iter_no_shuffle_fn": train_iter_no_shuffle_fn,
        "val_iter_fn": val_iter_fn,
        "test_iter_fn": test_iter_fn,
        "train_size": TRAIN_SIZE,
        "val_size": val_samples,
        "test_size": VAL_SIZE - val_samples,
        "image_shape": (target_height, target_width, 3),
        "num_classes": NUM_CLASSES,
        "class_names": _class_names(),
    }
    return bundle


def get_iterators(batch_size: int, *, seed: int = 0, **kwargs):
    bundle = get_dataset(batch_size, seed=seed, **kwargs)
    return bundle["train_iter_fn"](), bundle["val_iter_fn"](), bundle["test_iter_fn"]()
