#!/usr/bin/env python3
"""Backend-neutral Keras definition for the SANPO benchmark candidate."""

from __future__ import annotations

from typing import Any


ARCHITECTURE_REVISION = "lraspp_sigmoid_no_pooled_bn_v1"
DETAIL_ENDPOINTS = {
    4: "expanded_conv_project_bn",
    # Preserve the historical P0/P1-A endpoint for an isolated comparison.
    8: None,
}


def _dilated_output_stride_16_backbone(keras: Any, backbone: Any) -> Any:
    """Keep the deepest MobileNetV3 features at OS16 using atrous depthwise blocks."""
    dilated_depthwise = {
        "expanded_conv_8_depthwise",
        "expanded_conv_9_depthwise",
        "expanded_conv_10_depthwise",
    }

    def clone_layer(layer: Any) -> Any:
        config = layer.get_config()
        if layer.name == "expanded_conv_8_depthwise_pad":
            config["padding"] = ((0, 0), (0, 0))
        elif layer.name in dilated_depthwise:
            config["strides"] = (1, 1)
            config["padding"] = "same"
            config["dilation_rate"] = (2, 2)
        return layer.__class__.from_config(config)

    dilated = keras.models.clone_model(
        backbone,
        input_tensors=backbone.input,
        clone_function=clone_layer,
    )
    # Stride/dilation/padding do not change kernel shapes, so ImageNet weights
    # transfer exactly while the deepest spatial output changes from OS32 to OS16.
    dilated.set_weights(backbone.get_weights())
    return dilated


def build_mobilenetv3_lraspp(
    keras: Any,
    input_size: int,
    num_classes: int = 4,
    backbone_weights: str | None = None,
    *,
    backbone_alpha: float = 0.75,
    decoder_channels: int = 96,
    detail_output_stride: int = 8,
    semantic_output_stride: int = 32,
) -> Any:
    """Build the single authoritative MobileNetV3Small + Lite R-ASPP graph."""
    if input_size not in {256, 384, 512}:
        raise ValueError("input_size must be one of: 256, 384, 512")
    if backbone_alpha not in {0.75, 1.0}:
        raise ValueError("backbone_alpha must be one of: 0.75, 1.0")
    if decoder_channels <= 0:
        raise ValueError("decoder_channels must be positive")
    if detail_output_stride not in DETAIL_ENDPOINTS:
        raise ValueError("detail_output_stride must be one of: 4, 8")
    if semantic_output_stride not in {16, 32}:
        raise ValueError("semantic_output_stride must be one of: 16, 32")
    inputs = keras.Input(shape=(input_size, input_size, 3), dtype="float32", name="rgb")
    normalized = keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rgb_0_255_to_mobilenet")(inputs)
    backbone = keras.applications.MobileNetV3Small(
        input_tensor=normalized,
        include_top=False,
        weights=backbone_weights,
        alpha=backbone_alpha,
        minimalistic=False,
        include_preprocessing=False,
    )
    if semantic_output_stride == 16:
        backbone = _dilated_output_stride_16_backbone(keras, backbone)
    candidates: dict[int, tuple[Any, str]] = {}
    for layer in backbone.layers:
        shape = layer.output.shape
        if len(shape) == 4 and shape[1] is not None and shape[1] == shape[2]:
            candidates[int(shape[1])] = (layer.output, layer.name)
    high_size = input_size // semantic_output_stride
    low_size = input_size // detail_output_stride
    high = backbone.output
    configured_low_endpoint = DETAIL_ENDPOINTS[detail_output_stride]
    if configured_low_endpoint is None:
        low, low_endpoint = candidates[low_size]
    else:
        low_endpoint = configured_low_endpoint
        low = backbone.get_layer(low_endpoint).output
    if int(high.shape[1]) != high_size or int(low.shape[1]) != low_size:
        raise RuntimeError(
            "LR-ASPP endpoint contract mismatch: "
            f"semantic={tuple(high.shape)} expected OS{semantic_output_stride}; "
            f"detail={tuple(low.shape)} expected OS{detail_output_stride}"
        )
    context = keras.layers.GlobalAveragePooling2D(keepdims=True, name="lraspp_context_pool")(high)
    context = keras.layers.Conv2D(decoder_channels, 1, use_bias=False, name="lraspp_context_project")(context)
    # LR-ASPP uses the pooled branch as a bounded SE-style gate.  Both the
    # MobileNetV3 paper and torchvision reference implementation apply a
    # sigmoid directly after the 1x1 projection; pooled BatchNorm is unstable
    # for small batches and ReLU6 does not provide a [0, 1] gate.
    context = keras.layers.Activation("sigmoid", name="lraspp_context_sigmoid")(context)
    context = keras.layers.UpSampling2D(size=(high_size, high_size), interpolation="nearest", name="lraspp_context_up")(context)
    high = keras.layers.Conv2D(decoder_channels, 1, use_bias=False, name="lraspp_high_project")(high)
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
    model = keras.Model(inputs=inputs, outputs=logits, name="mobilenetv3_lraspp_4class")
    model.sanpo_feature_contract = {
        "architecture_revision": ARCHITECTURE_REVISION,
        "semantic_output_stride": semantic_output_stride,
        "semantic_endpoint": "deepest_backbone_output",
        "detail_output_stride": detail_output_stride,
        "detail_endpoint": low_endpoint,
        "gate": "global_pool_conv_sigmoid_no_pooled_batchnorm",
    }
    return model
