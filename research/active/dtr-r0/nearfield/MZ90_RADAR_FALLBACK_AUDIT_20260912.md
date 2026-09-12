# MZ90 Radar fallback provenance audit

Decision: prioritize observable spatial localization diagnosis over additional
persistence filtering. This is posthoc analysis of consumed MZ90 evidence, not a
new experiment, policy, promotion or fresh confirmation. No predictions or source
parameters were changed; no classifier or threshold was fitted.

## Attribution

For each sealed joint_spatial alert without valid ToF, inspect qualifying Radar
returns in the current frame, or the preceding frame for hysteresis sustain.
Labels are evaluator-only. Categories describe available support, not a causal
ablation: the existing two-of-three activation can involve different objects.

| Available support | TP frames | FP frames |
|---|---:|---:|
| Real objects only | 189 | 78 |
| Persistent phantom only | 7 | 44 |
| Transient clutter only | 0 | 1 |
| Mixed origins | 6 | 3 |
| Total | 202 | 126 |

194 TP frames have at least one supporting real object that is hazardous at the
alert timestamp. The other8 are coincident correct frame alerts without such
support. All126 FP have no hazardous real object by the frozen scene truth.
The78 real-only FP concern real but nonhazardous objects; this audit does not
equate every such case with a single angular error mechanism.

ToF packet loss accounts for36 TP and10 FP; the remaining166 TP and116 FP have
received packets without a selected valid return. Neither is clearance evidence.

## What observable histories establish

The existing three-frame qualifying-count median is3 for both TP and FP.
Most-negative radial velocity has the same five-number summary in both groups:
[-1.0,-0.8,-0.7,-0.6,-0.4]m/s. Minimum absolute stabilized bearing medians are
6.69deg TP and9.38deg FP, with substantial overlapping ranges. These are
descriptions of all qualifying supports, not fitted separators or track estimates.
They justify investigating spatial localization; they do not demonstrate that a
new angular threshold or tracker will improve the policy.

The persistent phantom uses exactly a static real point's noiseless Radar
kinematics: z=z0-0.7t, range=hypot(x,z), angle=atan2(x,z)-sensorYaw and
radial_velocity=-0.7z/range. Shared noise/quantization follows. Therefore motion
consistency alone cannot guarantee rejection of phantoms while keeping all
stationary obstacles. A separate read-only review confirmed this algebra.

This is overlapping finite observation support, **not identical observation
distributions**: real returns are sampled with probability0.8, while in-FOV
phantoms are always emitted before shared closest-four truncation. Real objects
can yield ToF; phantoms cannot. Longer histories may aid statistical distinction.
No global impossibility claim or real-sensor frequency estimate follows.

## Verification and retained evidence

Artifact: `artifacts.local/work/mz90-radar-fallback-audit-20260912/run-v1`.
`rows.json` retains all328 attributed frames and causal descriptors;
`summary.json` retains counts/quantiles; `evaluator-only-radar-origins.npz`
retains labels. The audit verifies22 original receipt hashes, replays the exact
sealed source with label-only instrumentation, and requires equality for every
raw observation and evaluator array, including NaNs. Numeric stable sorting and
all random draws are preserved.480 phantom timestamp checks verify radial
kinematic equality. Original predictions remain sealed and unchanged.

Reproduce with the project NumPy environment:

```text
python research/active/dtr-r0/nearfield/mz90_radar_fallback_audit.py --source-run artifacts.local/work/mz90-observable-contract-20260912/run-v1 --output artifacts.local/work/mz90-radar-fallback-audit-20260912/reproduction
```

Next bounded hypothesis, not executed here: determine whether causal multi-frame
range/bearing/velocity localization with explicit angular uncertainty can reject
nonhazardous real returns while preserving hazardous supports. Compare to frozen
matched hold, retain UNKNOWN, and keep persistent-phantom performance separate.
Do not optimize away synthetic phantoms using scene IDs, periodic flags, source
identity, detection-probability artifacts or evaluator labels. The present audit
ends here; no tracker, learning run or successor panel was launched.
