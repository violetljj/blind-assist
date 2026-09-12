# MZ82: explicit temporal visual collision evidence canary

Mode: `EXPLORE_POSTHOC_DIAGNOSTIC` on the consumed MZ77 Development source.

## Capability question

After MZ81 closed the frozen single-frame RGB residual route, does causal
five-frame image motion contain a low-false-positive ranking tail for collision
queries that the MZ79 range-capped ToF geometry expert misses?

This changes the information dimension, not model capacity. It trains no model,
uses no future frame, and creates no deployable threshold. The only primary
diagnostic is the forbidden posthoc ceiling:

`rescued true query bits at added FP <= 5`, separately for the three MZ79
5-klux typical profiles. A useful canary reaches at least 10 rescued bits in any
profile; 20 and 30 are reported as stronger descriptive landmarks. Zero in all
three profiles closes this explicit RGB-motion recipe and materially lowers the
priority of a learned temporal RGB expert. It does not rule out all video
representations or new physical sensors.

## Frozen observable-only predictor

Input is only the ordered 640x360 RGB frames, clip boundary, and nominal 0.1 s
sample interval already exposed to the MZ77 predictor. Frames are resized to
320x180 grayscale. Consecutive frames use fixed OpenCV Farneback dense optical
flow. In a central collision corridor, two overlapping vertical bands are used:

- HEAD: x 25--75%, y 15--55%;
- BODY: x 25--75%, y 40--95%.

For edge-bearing pixels in each band, the predictor records outward radial flow,
radial expansion rate, and the 25th-percentile positive image-space TTC. A
causal rolling median and support fraction over at most five frames supply the
stability term. BODY/HEAD near scores favor TTC at or below the frozen query end;
far scores favor TTC inside the corresponding near-to-far interval. Query ends
are the same fixed geometry definitions used by MZ80: 1.68/3.18 m for BODY and
1.63/3.13 m for HEAD. Nominal 1 m/s maps image TTC seconds to descriptive range
only inside this posed simulation.

The script writes and hashes `features.npz` and `predictor-receipt.json` before
opening evaluator labels. Feature constants and all score arithmetic are fixed
in source. No parameter, corridor, optical-flow setting, score weight, cutoff,
or method winner may be selected from MZ77 labels. Labels are used only for the
declared oracle ceiling and taxonomies.

## Baseline and limits

The baseline is frozen MZ79 `GEOMETRY_TEMPORAL` under its three 5-klux typical
hard-cap profiles. Rescue opportunities are known query bits where baseline ToF
is negative and the MZ80 profile-level observability gate is UNKNOWN. Existing
baseline positives are immutable. Missing or unknown evidence is never CLEAR.

MZ77 is four synthetic, settled-pose clips in one Willow scene. Target appearance,
ego motion, nominal speed, corridor geometry, and labels are not independent.
This canary is therefore a consumed-source mechanism diagnostic, not source-
separated confirmation, measured optical flow, natural video, latency, hardware,
alert, user-benefit, or safety evidence. A positive result only authorizes a new
source-separated temporal protocol; it does not promote a temporal expert.
