import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("KERAS_BACKEND", "jax")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

# import Dataset as dataset
import Dataset_foods as dataset
from model import MyCNN


ARTIFACT_DIR = Path("artifacts")
PARAMS_PATH = ARTIFACT_DIR / "cnn_params.msgpack"
PLOT_PATH = ARTIFACT_DIR / "sample_predictions.png"


def load_params(model: MyCNN, input_shape) -> Dict[str, Any]:
    rng = jax.random.PRNGKey(42)
    params_rng, dropout_rng = jax.random.split(rng)
    variables = model.init({"params": params_rng, "dropout": dropout_rng}, jnp.ones(input_shape), train=False)
    params = variables["params"]

    with PARAMS_PATH.open("rb") as f:
        params = serialization.from_bytes(params, f.read())
    return params


def main():
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Model parameters not found at {PARAMS_PATH}. Train the model first.")

    raw_config = OmegaConf.load("Config.yml")
    config = OmegaConf.to_container(raw_config, resolve=True)

    num_samples = 10
    bundle = dataset.get_dataset(batch_size=max(16, num_samples))
    test_iter = bundle["test_iter_fn"]()
    class_names = bundle.get("class_names")

    images_list = []
    labels_list = []
    while len(images_list) < num_samples:
        batch_images, batch_labels = next(test_iter)
        for img, lbl in zip(batch_images, batch_labels):
            images_list.append(img)
            labels_list.append(int(lbl))
            if len(images_list) == num_samples:
                break

    images = np.asarray(images_list)
    labels = np.asarray(labels_list, dtype=np.int32)

    model = MyCNN(config=config)
    # input_shape = (1, *config["InputShape"])
    input_shape = (1, config["InputShape"][0])
    params = load_params(model, input_shape)

    logits = model.apply({"params": params}, images, train=False)
    predictions = np.array(jnp.argmax(logits, axis=-1))

    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= num_samples:
            ax.axis("off")
            continue

        ax.imshow(images[idx])
        ax.axis("off")
        true_label = class_names[int(labels[idx])] if class_names else str(int(labels[idx]))
        pred_label = class_names[int(predictions[idx])] if class_names else str(int(predictions[idx]))
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with PLOT_PATH.open("wb") as f:
        f.write(buf.getvalue())

    print(f"Saved prediction grid to {PLOT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
