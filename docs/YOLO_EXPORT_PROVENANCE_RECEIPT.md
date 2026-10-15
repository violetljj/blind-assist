# YOLO TFLite export provenance receipt

Status: receipt contract; actual bundled-model provenance remains incomplete.

Schema: [blindassist_yolo_export_receipt_v1.schema.json](../tools/contracts/schemas/blindassist_yolo_export_receipt_v1.schema.json)

Scope: partial implementation of issue [#20](https://github.com/violetljj/blind-assist/issues/20).

This contract records evidence assertions for an export attempt. Validation
checks their shape and internal consistency; it does not execute an export,
read the referenced checkpoint/lock/fixture files, authenticate their contents,
or certify that an asserted experiment occurred. Every committed example is
synthetic. The positive fixture borrows the public target hash solely to exercise
the external-pin comparison; its other identities and measurements are invented
fixture data, never provenance evidence for the bundled model.

## Independent verdicts

- `byte_reproducibility`: the target hash is loaded independently from
  [config/public_release_assets.json](../config/public_release_assets.json).
  A receipt cannot choose its own pin. PASS also requires checkpoint URL/project,
  immutable revision, source/artifact matching hashes, license evidence, exporter
  URL/commit/script/environment, a lock path/hash, a nonempty versioned package
  inventory, and nonempty paths, positive sizes and hashes for all three artifacts.
  Missing evidence yields UNKNOWN; known identity contradictions yield FAIL even
  when other evidence is missing.
- `structural_equivalence`: requires matching valid inventories with exactly one
  input and one output, unique identities, named floating-point tensors and fixed
  positive shapes. Input shape is `[batch, imgsz, imgsz, 3]` in NHWC float32; output
  is rank-three float32 with the same batch. Parameters bind those dimensions.
  The v1 export uses half-precision weights, no NMS, no INT8 and no dynamic shapes;
  tensors themselves are unquantized (`quantization: null`). Preprocessing,
  postprocessing and metadata are bound by SHA-256 of their contract artifacts.
  Missing contract hashes yield UNKNOWN; contradictory tensors/parameters FAIL.
- `numerical_equivalence`: requires a hashed frozen fixture set and reference
  runtime, a positive denominator, finite nonnegative bound/errors, and an exact
  unique output inventory bound to both expected and observed tensor names.

The dependency-free executable enforces the keywords used by the committed
schema, then the cross-field rules. It is not a general JSON Schema engine.
Malformed JSON/schema fields are INVALID (nonzero exit), distinct from a valid
receipt truthfully reporting FAIL or UNKNOWN. In particular, non-finite JSON
numbers cannot form valid measured evidence. A successful validator exit means
receipt validity; callers must inspect individual verdicts before making claims.
A numerical PASS never establishes byte identity or structural equivalence.

## Authority boundary and remaining work

All receipts deny authority for application behavior, accuracy, accessibility
effectiveness, product promotion and safety. This change replaces no model and
changes no exporter/runtime behavior.

The exact immutable source checkpoint, actual export lock and clean export
receipt for the distributed bytes remain unestablished. There is no committed
frozen numerical corpus/reference-output evidence for them. Issue #20 stays open
until those facts are evidenced; completing this contract does not satisfy the
whole issue. Cross-platform FlatBuffer byte reproducibility also remains unknown.

Run the focused suite and example validator:

```powershell
python -m unittest tools.contracts.test_validate_yolo_export_receipt
python tools/contracts/validate_yolo_export_receipt.py tools/contracts/fixtures/yolo-export-receipt-v1/positive.json
```
