# P3 R0.1 Frozen Asset Producer Lock

This lock freezes the five producers before any P3 frozen asset is materialized. Every producer requires an exact request schema, verifies all bound input hashes, refuses overwrite, and emits a separate output-hash receipt.

The only producer permitted to load a model is the frozen-disagreement producer, and it may load only the hash-bound parent A2 checkpoint after output absence, identity-lock, source-catalog and receipt checks pass. It cannot construct P3 or an optimizer and cannot access holdout identities or outcomes.

The required execution order is exclusion ledger, role identity lock/public manifest, parent-A2 disagreement, final train/validation manifests and class weights, sealed bundle, then aggregate coverage receipt. No producer may run until this lock is committed.
