# MZ84: source-new bidirectional ToF/radar complementarity falsifier

Mode: `EXPLORE`, zero training, new analytic simulation source.

## Question and gate

Does a fixed responsibility-separated late fusion exploit genuinely different
ToF and radar error sets, rather than merely inheriting a dominant radar expert?

The source contains exactly six predeclared stress families, four 20-frame
episodes each: weak-reflector thin poles/head bars, near off-corridor clutter,
frontal wall plus multipath ghost, low-radial-velocity lateral crossing, head
yaw/pitch apparent motion, and simultaneous multiple targets. Strong-light ToF
loss and short ToF gaps are embedded in the wall and multi-target families.

Promotion to a later IMU/state-estimator question requires all of:

1. generic frame F1 of fusion is strictly greater than both single experts;
2. at least one family has ToF-unique true frames and a different family has
   radar-unique true frames;
3. fusion FP is no more than two frames above the higher-F1 single expert;
4. on every positive episode, fusion first correct alert is no more than 0.2 s
   later than the earlier correct single-expert alert.

Failure stops learned fusion. Passing retains only the fixed late-fusion policy
as a controlled-simulation challenger; it is not hardware evidence.

## Independent source and sensor contracts

The source is generated from latent range, azimuth, lateral motion, height,
reflectivity, observability, and ghost variables. Evaluator generic hazard is
derived from present/future corridor intersection. Sensor predictors do not read
the evaluator arrays.

Radar exposes only `(range, radial_velocity, coarse_azimuth, valid)`. Its fixed
forward model applies RCS/range-dependent detection, 0.05 m and 0.1 m/s
quantization, range/angle/velocity noise, 10-degree angular cells, same-cell
0.35 m merging, and explicit ghost returns. The MZ83 expert remains unchanged:
range <3.18 m, radial velocity <=-0.35 m/s, |azimuth| <=20 degrees, two-of-three
activation, and two-empty-frame release.

ToF exposes fine angle, metric range, height band and `valid/UNKNOWN`. The fixed
expert uses a 12-degree collision corridor and 3.18 m range. For lateral motion,
two valid observations estimate angular rate and test one-second future corridor
intersection. Missing and strong-light returns remain UNKNOWN, never CLEAR.

Fixed fusion is:

`ToF_alert OR (ToF_UNKNOWN AND Radar_alert)`.

When ToF is valid and clear, its finer geometry vetoes coarse radar clutter or
ghosts. Radar-only rescue is always `GENERIC_FORWARD_HAZARD / HEIGHT_UNKNOWN`.
BODY/HEAD is emitted only from a valid positive ToF observation.

## Limits

This deliberately constructed source tests policy logic under named failure
directions. It does not estimate how frequently these conditions occur or how
real sensors respond. RCS, multipath, head motion, ToF sunlight failure, target
motion, packet gaps and noise are analytic proxies, not calibrated measurements.
No natural imagery, UE rendering, people, actual radar, actual ToF or IMU is used.

A positive result means only that the responsibility decomposition can exploit
complementary observability when those failure modes occur. It cannot justify
sensor purchase, learned fusion, alert timing, deployment, user benefit, or
safety claims without measured and source-separated evidence.
