# Explicit ToF model input adapter

[tof_model_adapter.py](tof_model_adapter.py) implements the representation repair
motivated by [MZ75](MZ75_RETURN_SLOT_CONTROL_RESULTS_20260911.md). Raw observed
packets remain in their reported target order. The opt-in model view places
usable returns first, in stable ascending distance order; unusable slots follow
in original order. Invalid model ranges become zero with validity false.

`adapt_packet(packet)` retains all 1..4 configured slots and all 16 or64 zones.
Target signal, sigma, reflectance and status follow exactly the same permutation,
including quality observed on unusable slots. Zone ambient, SPAD count and raw
target count remain zone-level fields. Every field has separate value and
availability arrays, so missing quality cannot become a confident zero.
`original_slot`, measurement identity, provenance and raw ordering are trace
metadata, not predictor features. The original packet retains raw invalid ranges
and timestamps; no raw packet is overwritten or relabelled as hardware truth.

`legacy_packet_inputs(packet)` and vectorized `legacy_batch_inputs(ranges, valid)`
bridge to the frozen MZ70 range/valid domain. They accept64 zones and one or two
configured slots, pad a one-slot configuration with an unavailable second slot,
and reject usable ranges beyond4m or unrepresentable positive float32 values.
The4m check is model compatibility, not a sensor reach guarantee.16-zone or
four-target configurations are rejected by this legacy bridge rather than
silently resized or truncated; the general packet view still supports them.
Callers must handle rejection as unsupported input, never clearance.

```python
from tof_model_adapter import adapt_packet, legacy_packet_inputs

# Choose a causally aligned packet first using the existing align_to_rgb helper.
model_view = adapt_packet(packet)
legacy_values, original_slot = legacy_packet_inputs(packet)
# legacy_values contains only ranges and valid for the unchanged frozen model.
signal, signal_observed = model_view.target_fields['signal_per_spad']
```

The view supplies observed quality to a future quality-aware consumer. The frozen
neural models still consume only their original inputs; this change does not add
quality training, quality probabilities, a noise simulator or real-device gains.
It also does not automatically change any frozen experiment or application path.
New runners can opt into the adapter explicitly and keep raw packets for audit.

Verification used no model inference or evaluator labels. Five adapter tests
plus eight packet tests passed. Checks include lone-slot movement, stable ties,
order-invariant valid range inputs, raw packet preservation, inverse recovery of
all target quality/status and availability, quality on invalid returns, empty
zones, one-target padding, unsupported shapes and float32 underflow rejection.

The engineering replay covered8,192 existing frames and six input conditions,
49,152 packet transformations in total. All40,960 original IDEAL/MERGE/DROP/
ALL_INVALID and CLOSEST profile frames remained byte-identical. The8,192 FAR
frames matched an independent scalar movement; their2,048 original HELD rows
matched the frozen MZ75 packed inputs exactly, including frame identities.
Valid-distance multisets and valid-zone coverage were preserved throughout.
The full check took0.832s; individual4,096-frame transformations took16.5–21.6ms
on CPU, excluding archive I/O and hash checks from those individual timings.

[Replay script](../../../../artifacts.local/work/tof-model-adapter-20260911/acceptance.py),
[final receipt](../../../../artifacts.local/work/tof-model-adapter-20260911/receipt-v2.json),
[test log](../../../../artifacts.local/work/tof-model-adapter-20260911/tests-v2.log),
[focused tests](test_tof_model_adapter.py).
