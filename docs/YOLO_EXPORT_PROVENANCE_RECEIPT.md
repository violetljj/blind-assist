# YOLO TFLite export provenance receipt

Status: draft / review required  
Schema: [blindassist_yolo_export_receipt_v1.schema.json](../schemas/blindassist_yolo_export_receipt_v1.schema.json)  
Scope: issue [#20](https://github.com/violetljj/blind-assist/issues/20)

This contract records evidence for an export attempt. It does not replace the
bundled model, change exporter or runtime behavior, or make a release claim.
All committed examples are synthetic and are not claims about the current
asset.

## Independent verdicts

- byte_reproducibility is PASS only when the generated TFLite SHA-256 equals
  the pinned public-asset SHA-256; source checkpoint, labels, exporter repository,
  Git object ID, script, lockfile, and environment identities must all be complete
  and the source/artifact checkpoint hashes must agree.
- structural_equivalence compares expected and observed tensor contracts.
- numerical_equivalence compares finite, non-negative errors on a frozen fixture
  set using a positive non-boolean denominator, hash-bound reference runtime, and
  frozen bound. Numerical output names must bind exactly and uniquely to the
  inspected output-tensor inventory.

The executable validator first enforces the complete JSON Schema surface,
including required and additional properties. Export parameters are a closed,
non-empty v1 contract; extensions require a new schema version.

A numerical PASS never implies byte identity. Missing checkpoint,
environment, tensor, fixture, runtime, or denominator evidence produces
UNKNOWN. An observed hash mismatch, tensor mismatch, non-finite result, or
bound violation produces FAIL. Neither result may be rescued by another
verdict.

## Authority boundary

Every receipt explicitly denies authority for application behavior, accuracy,
accessibility effectiveness, product promotion, and safety. Those claims need
separate evidence and governance.

## Remaining provenance gaps

The repository does not yet publicly establish the exact immutable checkpoint,
checkpoint hash, fully locked export environment, fixture corpus, reference
outputs, or whether cross-platform FlatBuffer byte identity is achievable.
These values must remain unknown until supported by evidence; they must not be
inferred from the model family or current asset.

Run the committed invariant suite:

    python -m unittest scripts.test_validate_yolo_export_receipt

