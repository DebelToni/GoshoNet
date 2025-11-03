from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct


@struct.dataclass
class TrainState:
    step: jnp.ndarray
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Any
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState
    dropout_rng: jax.Array

    @classmethod
    def create(cls, *, apply_fn, params, tx, dropout_rng):
        opt_state = tx.init(params)
        return cls(
            step=jnp.array(0, dtype=jnp.int32),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dropout_rng=dropout_rng,
        )

    def apply_gradients(self, *, grads, dropout_rng):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            dropout_rng=dropout_rng,
        )


def create_train_state(
    rng: jax.Array,
    model,
    input_shape: Tuple[int, ...],
    learning_rate: float,
) -> TrainState:
    params_rng, dropout_init_rng = jax.random.split(rng)
    variables = model.init(
        {"params": params_rng, "dropout": dropout_init_rng}, jnp.ones(input_shape), train=True
    )
    params = variables["params"]
    tx = optax.adamw(learning_rate=learning_rate)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, dropout_rng=dropout_init_rng
    )
    return state


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


@jax.jit
def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    images, labels = batch
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images, train=True, rngs={"dropout": dropout_rng})
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
    metrics = {
        "loss": loss,
        "accuracy": compute_accuracy(logits, labels),
    }
    return state, metrics


@jax.jit
def eval_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    images, labels = batch
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)
    logits = state.apply_fn({"params": state.params}, images, train=False)
    loss = cross_entropy_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
    return {"loss": loss, "accuracy": accuracy}


def summarize_metrics(metrics: List[Dict[str, jnp.ndarray]]) -> Dict[str, float]:
    if not metrics:
        return {"loss": float("nan"), "accuracy": float("nan")}

    summary = {}
    for key in metrics[0]:
        values = [float(jax.device_get(item[key])) for item in metrics]
        summary[key] = sum(values) / len(values)
    return summary
