from typing import Any, Dict, List, Tuple

import flax.linen as nn


class Flatten(nn.Module):
    def __call__(self, x):
        return x.reshape((x.shape[0], -1))


class MaxPool(nn.Module):
    window_shape: Tuple[int, int]
    strides: Tuple[int, int]

    def __call__(self, x):
        return nn.max_pool(x, window_shape=self.window_shape, strides=self.strides, padding="VALID")


class MyCNN(nn.Module):
    config: Dict[str, Any]

    def setup(self):
        layers: List[Any] = []

        for idx, layer_cfg in enumerate(self.config["Layers"]):
            layer_type = layer_cfg["Type"]

            if layer_type == "Conv":
                kernel_size = layer_cfg.get("KernelSize", 3)
                strides = layer_cfg.get("Strides", 1)
                padding = layer_cfg.get("Padding", "SAME")

                if isinstance(kernel_size, int):
                    kernel_size = (kernel_size, kernel_size)
                if isinstance(strides, int):
                    strides = (strides, strides)

                layers.append(
                    nn.Conv(
                        features=layer_cfg["Features"],
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        use_bias=True,
                        name=f"conv_{idx}",
                    )
                )

                if layer_cfg.get("Activation", False):
                    layers.append(nn.relu)

                pool_cfg = layer_cfg.get("MaxPool")
                if pool_cfg:
                    pool_size = pool_cfg.get("Size", 2)
                    pool_stride = pool_cfg.get("Stride", 2)

                    if isinstance(pool_size, int):
                        pool_size = (pool_size, pool_size)
                    if isinstance(pool_stride, int):
                        pool_stride = (pool_stride, pool_stride)

                    layers.append(
                        MaxPool(
                            window_shape=pool_size,
                            strides=pool_stride,
                            name=f"maxpool_{idx}",
                        )
                    )

            elif layer_type == "Flatten":
                layers.append(Flatten(name=f"flatten_{idx}"))

            elif layer_type == "Dense":
                layers.append(
                    nn.Dense(
                        features=layer_cfg["Units"],
                        use_bias=True,
                        name=f"dense_{idx}",
                    )
                )

                if layer_cfg.get("Activation", False):
                    layers.append(nn.relu)

                dropout_rate = layer_cfg.get("Dropout")
                if dropout_rate:
                    layers.append(nn.Dropout(rate=dropout_rate, name=f"dropout_{idx}"))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        self.layers = layers

    def __call__(self, x, *, train: bool = True):
        h = x
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                h = layer(h, deterministic=not train)
            else:
                h = layer(h)
        return h
