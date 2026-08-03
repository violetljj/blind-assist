# Motion occupancy A0.1 candidate CLI smoke result

Date: 2026-08-03

Status: `CANDIDATE_ONLY_CLI_SMOKE_PASS`

The committed CLI ran end to end on the first 30 frozen
`walking_halfsphere` RGB frames using only RGB, Freiburg 3 intrinsics, the
frozen UniDepth and RAFT checkpoints, and the frozen A0.1 model.

- Candidate-valid frames: 30/30
- JSONL rows: 30
- MP4: 30 frames, 10 FPS, 640x480, first frame decodes successfully
- Known-reference comparison opportunities: 261
- Maximum absolute probability delta versus the earlier frozen evaluator:
  `0.00008250909470919332`
- Mean absolute probability delta: `0.000006455903355472168`
- Nine additional CLI opportunities belonged to the evaluator's
  reference-UNKNOWN first frame; candidate-only output correctly retained them.

The tiny deltas are consistent with repeated GPU inference and confirm that the
runnable CLI implements the same candidate, not a separately tuned demo path.

Ignored outputs:

- `occupancy.jsonl` SHA-256:
  `4D2EEA5D7EE7E1924AD34FDC1F781E59114C716FD764E9CFAADB30EBC50B1EF7`
- `occupancy.mp4` SHA-256:
  `A358B09898DA94C938DD9EC1367F738AA539033E5153B7A51AEFFE2CEA9A0EF6`
- `summary.json` SHA-256:
  `E6EC94F280304B136EFE2B4C527353E8E1168320355C745BBF6D91FC72057712`

This is a functional smoke test, not new accuracy evidence. Final external
camera validation still requires calibrated intrinsics and controlled geometry
truth.
