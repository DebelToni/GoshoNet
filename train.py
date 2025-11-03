import argparse
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
from flax import serialization
from omegaconf import OmegaConf

# import Dataset_foods as dataset
import Dataset as dataset
from model import MyCNN
from train_utils import create_train_state, eval_step, summarize_metrics, train_step


ARTIFACT_DIR = Path("artifacts")
PARAMS_PATH = ARTIFACT_DIR / "cnn_params.msgpack"


def save_params(params, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    param_bytes = serialization.to_bytes(params)
    with path.open("wb") as f:
        f.write(param_bytes)


def evaluate_split(state, iterator, num_steps: int) -> Dict[str, float]:
    metrics = []
    for _ in range(num_steps):
        batch = next(iterator)
        metrics.append(eval_step(state, batch))
    return summarize_metrics(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10 with JAX/Flax")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--train-steps", type=int, default=None, help="Limit training steps per epoch")
    parser.add_argument("--val-steps", type=int, default=None, help="Limit validation steps")
    parser.add_argument("--test-steps", type=int, default=None, help="Limit test steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable shuffling/repeat in the training iterator"
    )
    parser.add_argument("--val-samples", type=int, default=None, help="Validation samples for streaming datasets")
    parser.add_argument("--target-height", type=int, default=None, help="Resize height for streamed images")
    parser.add_argument("--target-width", type=int, default=None, help="Resize width for streamed images")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=None,
        help="Shuffle buffer size for streamed datasets (controls memory usage)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_config = OmegaConf.load("Config.yml")
    config = OmegaConf.to_container(raw_config, resolve=True)

    batch_size = args.batch_size or config["BatchSize"]
    epochs = args.epochs or config["Epochs"]
    learning_rate = args.learning_rate or config["LearningRate"]

    dataset_kwargs = {"batch_size": batch_size, "seed": args.seed}
    if args.val_samples is not None:
        dataset_kwargs["val_samples"] = args.val_samples
    if args.target_height is not None:
        dataset_kwargs["target_height"] = args.target_height
    if args.target_width is not None:
        dataset_kwargs["target_width"] = args.target_width
    if args.shuffle_buffer is not None:
        dataset_kwargs["shuffle_buffer_size"] = args.shuffle_buffer

    data_bundle = dataset.get_dataset(**dataset_kwargs)
    print("Loaded dataset bundle", flush=True)

    train_iter_fn = data_bundle["train_iter_fn"]
    if args.no_shuffle and "train_iter_no_shuffle_fn" in data_bundle:
        train_iter_fn = data_bundle["train_iter_no_shuffle_fn"]
    val_iter_fn = data_bundle["val_iter_fn"]
    test_iter_fn = data_bundle["test_iter_fn"]

    train_iter = train_iter_fn()
    print("Created training iterator", flush=True)

    data_image_shape = tuple(data_bundle["image_shape"])
    input_shape = (1, *config["InputShape"])
    if tuple(config["InputShape"]) != data_image_shape:
        print(
            f"⚠ Config InputShape does not match dataset image shape. Using dataset shape for initialization. Shape {data_image_shape}",
            flush=True,
        )
        input_shape = (1, *data_image_shape)
    cfg_layers = config["Layers"]
    if cfg_layers and cfg_layers[-1]["Type"] == "Dense":
        final_units = cfg_layers[-1]["Units"]
        target_units = data_bundle.get("num_classes", final_units)
        if final_units != target_units:
            cfg_layers[-1]["Units"] = target_units
            print(
                f"Adjusted final dense units from {final_units} to match dataset classes {target_units}",
                flush=True,
            )

    model = MyCNN(config=config)
    rng = jax.random.PRNGKey(args.seed)
    state = create_train_state(rng, model, input_shape, learning_rate)
    print("Initialized model parameters", flush=True)

    train_steps = args.train_steps or (data_bundle["train_size"] // batch_size)
    val_steps = args.val_steps or max(1, data_bundle["val_size"] // batch_size)
    test_steps = args.test_steps or max(1, data_bundle["test_size"] // batch_size)
    print(f"Train steps {train_steps}, Val steps {val_steps}, Test steps {test_steps}", flush=True)

    if train_steps < 1:
        raise ValueError("train_steps must be at least 1")

    progress_interval = None

    for epoch in range(1, epochs + 1):
        train_metrics = []

        for step_idx in range(train_steps):
            batch = next(train_iter)
            state, metrics = train_step(state, batch)
            train_metrics.append(metrics)

            if progress_interval is None:
                progress_interval = max(1, train_steps // 5)

            if (
                step_idx == 0
                or (step_idx + 1) % progress_interval == 0
                or step_idx + 1 == train_steps
            ):
                print(
                    f"Epoch {epoch:02d} step {step_idx + 1}/{train_steps} complete",
                    flush=True,
                )
        train_summary = summarize_metrics(train_metrics)

        val_iter = val_iter_fn()
        val_summary = evaluate_split(state, val_iter, val_steps)

        print(
            f"Epoch {epoch:02d} "
            f"| train loss {train_summary['loss']:.4f}, acc {train_summary['accuracy']:.4f} "
            f"| val loss {val_summary['loss']:.4f}, acc {val_summary['accuracy']:.4f}",
            flush=True,
        )

    test_iter = test_iter_fn()
    test_summary = evaluate_split(state, test_iter, test_steps)
    print(
        f"Test    | loss {test_summary['loss']:.4f}, acc {test_summary['accuracy']:.4f}",
        flush=True,
    )

    save_params(state.params, PARAMS_PATH)
    print(f"Saved parameters to {PARAMS_PATH}", flush=True)


if __name__ == "__main__":
    main()
