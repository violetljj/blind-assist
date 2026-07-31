# D0-A successor R0: fixed-clip observation review

You are one isolated Agent reviewer in `CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR`.
Review only the frozen source images listed in the successor calibration manifest. Do not inspect
YOLO, candidate output, depth, segmentation, risk, feedback, previous labels, another review, or
the current result. The predeclared fixed clip boundaries are already final. Do not merge, split,
extend, or name a natural event.

## Observation labels

For every slot, emit exactly one of:

- `VISIBLE_CENTRAL_OBSTRUCTION_PRESENT`: a reviewable foreground or midground entity visibly
  blocks the background scene or terminates the central line of sight inside the frozen central ROI.
- `NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE`: the ROI is reviewable but no qualifying visible
  central obstruction is present. This does not mean clear, safe, or obstacle-free.
- `NOT_EVALUABLE`: blur, darkness, occlusion, turning/scene transition, title/edit frame, or
  another visibility failure prevents the observation from supporting either label.

Do not call a background building, ground, sky, water, distant figure, texture, or shadow an
obstruction by itself. ROI occupancy alone is insufficient.

## Quality and rationale

Use one quality state: `STABLE`, `TURNING`, `BLURRED`, `DARK`, `OCCLUDED`, or
`OTHER_NOT_EVALUABLE`. For each slot provide a short source-only rationale and one compact
`rationale_code`, such as `VISIBLE_FOREGROUND_BLOCKER`, `REVIEWABLE_BACKGROUND_ONLY`,
`TURNING_VIEW_NOT_EVALUABLE`, `DARK_NOT_EVALUABLE`, `BLUR_NOT_EVALUABLE`,
`OCCLUSION_NOT_EVALUABLE`, or `SCENE_CUT_NOT_EVALUABLE`.

## Output contract

Return JSON only, with no aggregate score and no event grouping:

```json
{
  "schema_version": "blindassist.central_obstruction_d0a_successor_review.v1",
  "protocol_id": "CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR",
  "evidence_instance": "CENTRAL_OBSTRUCTION_D0_A_SUCCESSOR_FIXED_CLIP_CALIBRATION_R0",
  "review_id": "unique-review-id",
  "review_context": "FRESH_ISOLATED_PRIMARY",
  "reviewer_type": "CODEX_AGENT",
  "candidate_output_visible": false,
  "prior_review_visible": false,
  "other_review_visible_before_submission": false,
  "source_only_view": true,
  "observations": [
    {
      "unit_id": "frozen unit id from manifest",
      "slot_ordinal": 0,
      "label": "one frozen label",
      "quality_state": "one frozen quality state",
      "rationale_code": "one compact code",
      "rationale": "one short visual reason"
    }
  ]
}
```

Cover every frozen slot exactly once, in ascending `unit_id` and `slot_ordinal` order. The
validator will reject missing, duplicate, or extra slots. Do not report a unit-level state; the
program will derive it deterministically after both raw observation reviews are sealed.
