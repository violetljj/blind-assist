#!/usr/bin/env python3
"""Backend-neutral Keras definition for the SANPO benchmark candidate."""

from __future__ import annotations

from typing import Any


def build_mobilenetv3_lraspp(
    keras: Any, input_size: int, num_classes: int = 4, backbone_weights: str | None = None,
) -> Any:
    """Build the single authoritative MobileNetV3Small + Lite R-ASPP graph."""
    inputs = keras.Input(shape=(input_size, input_size, 3), dtype="float32", name="rgb")
    normalized = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rgb_0_255_to_mobilenet")(inputs)
    backbone = keras.applications.MobileNetV3Small(
        input_tensor=normalized,
        include_top=False,
        weights=backbone_weights,
        alpha=0.75,
        minimalistic=False,
        include_preprocessing=False,
    )
    candidates: dict[int, Any] = {}
    for layer in backbone.layers:
        shape = layer.output.shape
        if len(shape) == 4 and shape[1] is not None and shape[1] == shape[2]:
            candidates[int(shape[1])] = layer.output
    # LR-ASPP's semantic branch must use the deepest backbone feature.  Using
    # ``max(... <= input/16)`` accidentally selected the shallow 1/16 tensor
    # and pruned the rest of MobileNetV3 from the functional graph.
    # Ignore squeeze/excitation 1x1 tensors; the deepest spatial feature map is
    # the 1/32 (8x8 at 256px) backbone output.
    high_size = min(size for size in candidates if size > 1)
    low_size = max(size for size in candidates if size <= input_size // 8)
    high = candidates[high_size]
    low = candidates[low_size]
    context = keras.layers.GlobalAveragePooling2D(keepdims=True, name="lraspp_context_pool")(high)
    context = keras.layers.Conv2D(96, 1, use_bias=False, name="lraspp_context_project")(context)
    context = keras.layers.BatchNormalization(name="lraspp_context_bn")(context)
    context = keras.layers.ReLU(max_value=6.0, name="lraspp_context_relu")(context)
    context = keras.layers.UpSampling2D(size=(high_size, high_size), interpolation="nearest", name="lraspp_context_up")(context)
    high = keras.layers.Conv2D(96, 1, use_bias=False, name="lraspp_high_project")(high)
    high = keras.layers.BatchNormalization(name="lraspp_high_bn")(high)
    high = keras.layers.ReLU(max_value=6.0, name="lraspp_high_relu")(high)
    high = keras.layers.Multiply(name="lraspp_context_gate")([high, context])
    high = keras.layers.UpSampling2D(size=(low_size // high_size, low_size // high_size), interpolation="nearest", name="lraspp_high_up")(high)
    low = keras.layers.Conv2D(32, 1, use_bias=False, name="lraspp_low_project")(low)
    low = keras.layers.BatchNormalization(name="lraspp_low_bn")(low)
    low = keras.layers.ReLU(max_value=6.0, name="lraspp_low_relu")(low)
    fused = keras.layers.Concatenate(name="lraspp_fuse")([high, low])
    logits = keras.layers.Conv2D(num_classes, 1, name="semantic_logits")(fused)
    scale = input_size // low_size
    logits = keras.layers.UpSampling2D(size=(scale, scale), interpolation="bilinear", name="semantic_logits_256")(logits)
    return keras.Model(inputs=inputs, outputs=logits, name="mobilenetv3_lraspp_4class")
