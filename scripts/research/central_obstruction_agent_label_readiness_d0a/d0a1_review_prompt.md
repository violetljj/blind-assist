# D0-A1 central-image obstruction observation prompt

Review only the frozen RGB observations and the normalized
`CENTRAL_IMAGE_ATTENTION_REGION`. Do not infer intended route, free space,
traversability, collision risk, safety, or what happens outside the image.

For every observation emit exactly one label:

- `VISIBLE_CENTRAL_OBSTRUCTION_PRESENT`: a visible physical person, object,
  barrier, wall/surface, or other scene element clearly occupies or blocks a
  material part of the frozen ROI. This is an image observation, not a risk or
  collision claim.
- `NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE`: the ROI is reviewable and no
  clearly visible central obstruction is supported. This never means clear,
  safe, traversable, or obstacle-free.
- `NOT_EVALUABLE`: the ROI observation cannot be made reliably because of
  extreme blur/darkness, title card or non-scene content, hard scene cut,
  unusable native dimensions, near-total occlusion, or camera orientation that
  removes the intended forward-looking image content.

Also emit one camera-quality state:
`STABLE`, `TURNING`, `BLURRED`, `DARK`, `OCCLUDED`, or
`OTHER_NOT_EVALUABLE`.

Parent events are maximal consecutive observations with the same label inside
one frozen clip. A scene cut, session boundary, non-consecutive observation,
or label change always closes the event. Do not bridge a disagreement or
`NOT_EVALUABLE` observation. Preserve all raw observation labels and event
boundaries.

Candidate-model output, YOLO, segmentation, depth, risk, feedback, prior
review labels, and another review pass are forbidden inputs.
