# Bonn A0.1 reference correction

Date: 2026-08-03

Status: `FROZEN_BEFORE_ANY_KNOWN_BONN_OCCUPANCY_OPPORTUNITY`

The first protocol assumed the Bonn pose world vertical was its z axis. The
first source execution produced 140/150 paired-valid candidate frames but zero
known sensor clearance opportunities: the transformed z-plane was nearly
orthogonal to the sensor-depth ground plane. No frozen occupancy probability
could be evaluated, so that execution is `NOT_EVALUABLE`, not a model outcome.

The reference-only first 30 frames were recalibrated as a general world plane
`normal dot point + offset = 0`. Camera-frame RANSAC planes were transformed by
the supplied camera poses and robustly aggregated. The corrected frozen planes
are:

- tracking normal `[-0.09003391161250643, -0.9210380300635856,
  -0.3789232665545108]`, offset `1.5700314257061778 m`;
- tracking2 normal `[-0.1328921300754135, -0.9215256070712604,
  -0.3648701649572957]`, offset `1.4000356415239954 m`.

Their offset MAD values are 0.03832 m and 0.02780 m. Evaluation windows,
candidate inputs, model, bands, horizons, gates, and all other protocol fields
remain unchanged. This correction is frozen before rerunning either source and
before any Bonn occupancy probability exists.
