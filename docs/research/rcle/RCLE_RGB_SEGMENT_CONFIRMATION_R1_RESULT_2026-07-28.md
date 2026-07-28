# RCLE RGB Segment Confirmation R1 result

## Terminal

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL`

The two frozen real-segment RGB identities did not close under their signed,
one-shot, budgeted opaque-extraction attempts. Therefore no final RGB
identity/synchronization lock was issued, no pixel was decoded, and the frozen
RGB algorithm was not run.

This is a valid `NOT_EVALUABLE` terminal. It is not evidence that the RGB
algorithm succeeds or fails.

Authoritative terminal:

- `artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/protocol_terminal.v1.json`
- SHA-256:
  `03fbac1d815072639b00393cb058f31aacba5de0b0a270c6d440f2e0bab10753`

## Locked segment dispositions

| frozen segment | role | immutable terminal | RGB frames | disposition |
|---|---:|---|---:|---|
| `OPENLORIS_CORRIDOR / corridor1-1:w004` | positive | `INVALID_IDENTITY_EXTRACTION_CLOSE_ATTEMPT / URLError` | 0 | identity not closed; claim consumed; retry forbidden |
| `DLR_RGBD_VICON / extreme_geometry/hexagon_01:w001` | below | `SEGMENT_IDENTITY_NOT_EVALUABLE / DLR_BYTE_BUDGET_EXHAUSTED_OR_RGB_GUARD_ABSENT` | 0 | identity not closed; claim consumed; retry forbidden |

The DLR transport ledger contains 129 contiguous, unique, attempt-1 HTTP 206
ranges. Their byte sum is `1,065,353,305`, exactly matching the terminal
`remote_bytes` and remaining below the frozen `1,073,741,824`-byte cap. This
proves the bounded transport accounting, not RGB identity closure.

Both terminal receipts received independent review:

- OpenLORIS:
  `openloris_identity_terminal_independent_review.v1.json`,
  SHA-256
  `99830376dcba266f607df636fd92e44c5f6164856d0d599b998f171fbe657cd9`
- DLR:
  `dlr_identity_terminal_independent_review.v1.json`,
  SHA-256
  `d56edd0d6dd1afed4e652fd4c0ce4a3ad92f577e9e22628b356ad43e0b58c69f`

## Ledgers and alignment

Because there are zero identity-eligible RGB frames, the frame ledger contains
zero per-frame observations and two explicit segment zero-frame terminal rows.
No missing frame is imputed or converted into a negative observation.

- `frame_ledger.v1.jsonl`: zero eligible frames and zero frame rows
- `abstention_ledger.v1.jsonl`: the two segment abstentions plus the
  protocol-level execution abstention
- `alignment_metrics.v1.json`: pair denominator `0`; all alignment values
  `null`, not zero

Pixel decode calls and RGB algorithm calls are both `0`.

## Claim ceiling

OpenLORIS-positive and DLR-below remain source-role confounded. No
positive/below comparison was performed, and this terminal grants no mechanism,
discrimination, performance, or generalization evidence.

MVSEC was not accessed because its exact RGB capture identity and synchronization
relation were not confirmed before the R1 terminal.

The next stages remain closed:

1. RGB segment mechanism validation: `NOT_EVALUABLE`
2. Independent or extended performance qualification: `NOT_AUTHORIZED`
3. Host offline replay: `NOT_AUTHORIZED`
4. Android, product, and safety conclusions: `NOT_AUTHORIZED`

The consumed claims must not be retried, replaced, widened, or rescued by a
full-source download.
