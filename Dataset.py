from typing import Callable, Dict, Iterator, Tuple

import keras
import numpy as np

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

VAL_SPLIT = 500


def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    val_images, val_labels = train_images[:VAL_SPLIT], train_labels[:VAL_SPLIT]
    train_images, train_labels = train_images[VAL_SPLIT:], train_labels[VAL_SPLIT:]
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    return np.squeeze(labels.astype(np.int32), axis=-1)


def _make_iterator_from_arrays(
    images: np.ndarray, labels: np.ndarray, batch_size: int, *, repeat: bool, shuffle: bool, seed: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    images = preprocess_images(images)
    labels = preprocess_labels(labels)
    num_samples = images.shape[0]

    if num_samples < batch_size:
        raise ValueError("Batch size larger than dataset size")

    rng = np.random.default_rng(seed)

    def generator():
        while True:
            if shuffle:
                indices = rng.permutation(num_samples)
            else:
                indices = np.arange(num_samples)

            for start in range(0, num_samples - batch_size + 1, batch_size):
                batch_idx = indices[start : start + batch_size]
                yield images[batch_idx], labels[batch_idx]

            if not repeat:
                break

    return generator()


def get_dataset(batch_size: int, *, seed: int = 0, **_) -> Dict[str, object]:
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_cifar10()

    train_iter_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _make_iterator_from_arrays(
        train_images, train_labels, batch_size, repeat=True, shuffle=True, seed=seed
    )
    train_iter_no_shuffle_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _make_iterator_from_arrays(
        train_images, train_labels, batch_size, repeat=True, shuffle=False, seed=seed
    )
    val_iter_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _make_iterator_from_arrays(
        val_images, val_labels, batch_size, repeat=False, shuffle=False, seed=seed
    )
    test_iter_fn: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]] = lambda: _make_iterator_from_arrays(
        test_images, test_labels, batch_size, repeat=False, shuffle=False, seed=seed
    )

    bundle = {
        "train_iter_fn": train_iter_fn,
        "train_iter_no_shuffle_fn": train_iter_no_shuffle_fn,
        "val_iter_fn": val_iter_fn,
        "test_iter_fn": test_iter_fn,
        "train_size": int(train_images.shape[0]),
        "val_size": int(val_images.shape[0]),
        "test_size": int(test_images.shape[0]),
        "image_shape": tuple(map(int, train_images.shape[1:])),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
    }
    return bundle


def get_iterators(batch_size: int, *, seed: int = 0, **kwargs):
    bundle = get_dataset(batch_size, seed=seed, **kwargs)
    return bundle["train_iter_fn"](), bundle["val_iter_fn"](), bundle["test_iter_fn"]()
