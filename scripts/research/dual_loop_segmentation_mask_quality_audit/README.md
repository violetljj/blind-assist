# Segmentation mask quality audit

This package is a read-only, frame-wise QA sidecar for existing RGB/mask
pairs. It is not a mask-redrawing or relabeling tool.

The original mask is always retained by path, file SHA-256, decoded-array
SHA-256, and frame identity. An optional model output is stored as
`authority=PROPOSAL_ONLY`; it is compared for review signals only. No command
in this package writes to an input RGB, original mask, or proposal file, and
no proposal is written into `mask_path`.

## Command

From the repository root:

```powershell
& E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.dual_loop_segmentation_mask_quality_audit.audit `
  --manifest artifacts.local/evidence/<bundle>/mask_qa_manifest.jsonl `
  --base-root . `
  --config configs/dual_loop_segmentation_mask_quality_audit/default.json `
  --review-file artifacts.local/evidence/<bundle>/visual_review.jsonl `
  --output-root artifacts.local/evidence/<bundle>/mask-quality-audit-r0
```

The output directory must be new or empty. It contains:

- `summary.json`: overall status, per-frame counts, reason-code counts, and
  provenance;
- `frame_results.jsonl`: one `PASS`, `REVIEW`, or `INVALID` result per frame;
- `review_queue.json`: non-PASS frames for visual review.

`--print-reason-codes` prints the machine-readable code catalog without
reading dataset files.

## Manifest contract

Each JSONL row must include `id`, `session_id`, `sequence_id` (recommended),
`frame_id`, `rgb_path`, `mask_path`, `rgb_sha256`, `mask_sha256`, `label_space`,
`class_order`, and `source_to_expected_mapping`. Paths are relative to
`--base-root` unless absolute. `rgb_frame_key` and `mask_frame_key` must be
equal, or a single explicit `pairing_key` must bind the pair. A model sidecar may add
`proposal_mask_path` and, if needed, `proposal_mask_decoder`.

The default contract is the current RISKSEG order:

```text
0 walkable
1 blocking_obstacle
2 boundary_level_change
3 unknown_nonwalkable
```

If an old mask uses the historical obstacle/boundary order, create a new,
hash-closed derived view with the explicit `1 -> 2, 2 -> 1` mapping. Do not
change the original file or silently reinterpret it in this audit.

## Human review

`require_visual_review=true` means a technical decode alone can never produce
`PASS`. The first visual pass should inspect only RGB and the original mask;
keep model proposals hidden. A review sidecar is JSONL, for example:

```json
{"id":"session-0001","status":"PASS","reason_codes":[],"reviewer_id":"reviewer-a"}
{"id":"session-0002","status":"INVALID","reason_codes":["OBSTACLE_BOUNDARY_SWAP_SUSPECTED"],"reviewer_id":"reviewer-a"}
```

`REVIEW` is retained evidence that requires inspection; it is not eligible for
training/truth evaluation. `INVALID` is fail-closed. A reviewer may mark a
semantic suspicion as `INVALID` when the RGB review confirms it; the original
mask remains unchanged in either case.
