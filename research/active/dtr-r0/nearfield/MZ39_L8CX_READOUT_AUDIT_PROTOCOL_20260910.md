# MZ39: audit ideal packets against a specific VL53L8CX constraint

2026-09-10 EXPLORE diagnostic. Target sensor for subsequent simulation is
VL53L8CX, as clarified by the user; the existing single-zone adapter does not
define this research input. Do not assume strong sensing capabilities.

Question: how often do saved MZ36 ideal two-return packets report a pair less
than 0.600 m apart? Such pairs are not justified as separately detected targets
by UM3109 Rev12 section4.10, which states a600mm minimum separation. This is
one concrete mismatch check, not a complete sensor simulator or task metric.

Baseline: untouched MZ36 inference-v1/predictions.npz,380 admitted frames,
8x8 zones, two return slots. No model outcomes or evaluator event labels are
needed. Saved zeros in invalid slots remain invalid, never free-space evidence.

Execute one CPU saved-array audit, zero inference/training/capture. Bind the
packet archive, original receipt and multizone64_observation.py by SHA256;
read the receipt before scoring, and preserve all frozen files. Report total
zones,0/1/2-valid-return counts, frames containing dual returns, pair separation
quantiles, pairs/frames below0.600m, and pairs below one documented10cm bin.
Retain exact violating frame/zone identities. Float32 ranges are assessed as
their stored values without rounding or tolerance changes; report equality.

Verification: a standard-library loop independently recounts all zones and
the strict separation predicate; counts must reconcile with vectorized NumPy.
This checks the diagnostic implementation, not physical validity of surviving
pairs. No separate hardware measurement is available in this task.

Decision: any sub600mm pair confirms an exercised ideal-readout mismatch and
prioritizes a sensor-constrained successor before interpreting new-object
results as VL53L8CX capability. Zero pairs means only this constraint was not
exercised. In either case, unknown signal strength/reflectance, mixed returns,
ambient light, temporal filtering and target selection remain unvalidated.
Do not turn this audit into a claim about task performance after degradation,
discard existing results, or relabel ideal packets as physical observations.

Official source inspected in the current task:
https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf
Sections4.9/4.10 also document strongest/closest ordering, default strongest,
and configurable1..4targets with one target by default. Two slots are an
explicit configuration assumption, not a promise of two valid observations.

Budget/stop: one saved380-frame audit, compact JSON only. No acquisition,
fit, cutoff change, raw-array copy or follow-on sweep inside this experiment.

