# D0-A1 central-image obstruction observation prompt R1

Review only the frozen RGB observations and the normalized
`CENTRAL_IMAGE_ATTENTION_REGION`. Do not infer intended route, free space,
traversability, collision risk, safety, or what happens outside the image.

For every observation emit exactly one label:

- `VISIBLE_CENTRAL_OBSTRUCTION_PRESENT`: a foreground or midground physical
  person, vehicle, barrier, pole, close wall/surface, or other entity overlaps
  the ROI and visibly occludes scene content behind it or terminates a central
  line of sight. Background buildings, ground, sky, water, distant
  architecture, texture, shadows, and an object merely being visible inside
  the ROI do not satisfy this label.
- `NO_VISIBLE_CENTRAL_OBSTRUCTION_EVIDENCE`: the forward-looking ROI is
  reviewable and no foreground/midground entity meeting the positive rule is
  supported. This never means clear, safe, traversable, free space, or
  obstacle-free.
- `NOT_EVALUABLE`: the forward-looking ROI observation cannot be made reliably
  because of extreme blur/darkness, title card or non-scene content, a hard
  scene cut, unusable native dimensions, near-total occlusion, or camera
  orientation that removes forward-looking scene content.

Also emit one camera-quality state:
`STABLE`, `TURNING`, `BLURRED`, `DARK`, `OCCLUDED`, or
`OTHER_NOT_EVALUABLE`.

Parent events are maximal consecutive frozen observations with the same label
inside one clip. A clip/session boundary, hard scene cut, non-consecutive
observation, or label transition closes the event. Consecutive
`NOT_EVALUABLE` observations caused by the same continuous episode remain one
event; they are not bridged across another label or a scene cut. Preserve all
raw observation labels and event boundaries.

Candidate-model output, YOLO, segmentation, depth, risk, feedback, prior
review labels, and another review pass are forbidden inputs.
