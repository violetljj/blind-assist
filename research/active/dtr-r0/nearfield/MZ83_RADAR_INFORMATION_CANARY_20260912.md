# MZ83: coarse radar information canary

Mode: `EXPLORE`, consumed MZ77 controlled Development source, simulation only.

## Capability question and stop rule

Can a coarse, physically independent radar-like observation recover generic
forward collision hazards lost by MZ79 range-capped ToF without inheriting the
same optical observability failure?

Primary gate, separately for all three MZ79 5-klux typical profiles:

- rescue at least 25% of the frozen ToF generic-hazard false-negative frames;
- add at most five generic-hazard false-positive frames.

Passing only authorizes a separately bounded ToF+radar late-fusion experiment.
Failure closes this coarse radar recipe and stops radar-network/fusion escalation.
No radar hardware purchase, learned radar model, IMU project, or application
change follows automatically.

## Radar-like forward model

Only the sensor-simulation stage may read each MZ77 native scene-depth image.
It does not read the source case specification, target actor names, masks,
BODY/HEAD labels, query truth, or MZ79 predictions. It collapses a fixed vertical
antenna aperture into nine 10-degree azimuth cells over a 90-degree FoV and emits
at most two returns per cell. Surface range clusters require visible support.

The deterministic forward model includes:

- 0.25 m source range clustering, 0.05 m measured range quantization, and
  0.06 m Gaussian range noise;
- 10-degree angular cells, 5-degree measured azimuth quantization, and
  2-degree Gaussian azimuth noise;
- radial velocity derived from consecutive same-cell range clusters, then
  0.1 m/s quantization and 0.08 m/s noise;
- range-dependent missed detections with a fixed seed;
- at most two returns per azimuth cell and 0.35 m same-cell return merging;
- low-rate static/dynamic clutter over the sensor FoV;
- invalid radial velocity on the first frame of every clip.

The sealed predictor input is only `(range_m, radial_velocity_mps,
azimuth_deg, valid)`, clip boundary and nominal timestamp. Native depth is not
retained in the packet artifact.

## Fixed generic hazard expert

A current return is a candidate when range is below 3.18 m, radial velocity is
at most -0.35 m/s, and absolute azimuth is at most 20 degrees. Generic hazard
activates after two candidate frames in the last three and releases after two
consecutive empty frames. TTC is `range / -radial_velocity` from the closest
current candidate. All rules are causal and fixed before evaluator access.

Generic truth is any known MZ77 BODY/HEAD near/far collision query. The primary
comparison uses frame-level ToF generic hazard, formed by OR over the four frozen
MZ79 geometry-temporal query outputs. Existing ToF positives are immutable.

BODY/HEAD attribution is secondary and is not fabricated from radar. When ToF
has a query-level observation it retains its original attribution; radar-only
rescues remain `GENERIC_FORWARD_HAZARD / HEIGHT_UNKNOWN`.

## Evidence limits

The MZ77 approach is a posed 1 m/s synthetic trajectory with three simple added
targets and one no-added-target control in a single scene. Native render depth
has no radar cross section, material response, multipath, antenna pattern,
range-Doppler ambiguity, interference, or measured device calibration. The
fixed support/dropout/clutter model only prevents a direct truth feed; it is not
a calibrated 60 GHz simulator.

Therefore a pass means only that coarse range, closing velocity and azimuth are
informationally sufficient under this declared sensitivity model. It does not
establish real radar detection of walls, thin poles, head bars, people, or any
alert, hardware, deployment, user-benefit, or safety performance.
