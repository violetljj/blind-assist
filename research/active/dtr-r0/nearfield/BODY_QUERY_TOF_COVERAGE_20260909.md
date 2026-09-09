# Fixed ToF-footprint observation coverage

2026-09-09 EXPLORE; source-only, no sensor-return or fusion-performance claim.
Use all600 existing captured endpoints:120 original-fixture/background-translation
frames and480 simplified-fixture frames. Report all frames and the unchanged
admitted subsets separately. No source selection, recapture or model fitting.

Question: can the visible HEAD bar substantially fill a fixed, centered,
VL53L1X-sized observation footprint in these scenarios? Compare two declared
centered ideal square footprints with27degree and15degree DIAGONAL FoV. Camera
and virtual ToF optical centers/axes coincide. The angular square is a geometric
proxy, not a calibrated SPAD response, ROI optical model or actual installation.
Use the same fixed footprints on every frame; no target-guided aiming.

Read hash-bound scene and isolated-target axial native depths. Project the
footprint through the existing640x360,100degree horizontal camera calibration.
Weight rays by solid angle; count visible target support where isolated and scene
depth agree within0.03m. Preserve invalid native pixels as unknown. Separately
record support beyond4m; this is a nominal range cutoff, not measured validity.
Compute target coverage, missing support, and geometric target/background range
distributions. All target identities, masks and full depth remain evaluator-only.

Predeclared geometric dominance marker: >=95% of the whole footprint is visible
target within4m, with<=1% missing native support. It is an engineering screen for
a nearly homogeneous target footprint, NOT sensor validity or detectability.
Any target/non-target mixture remains explicitly mixed even if dominance holds.
Never translate low target coverage into proven ToF failure, or dominance into
proven success. Nearest, center, median and mean depth are not simulated returns.

Stop after one CUDA coverage pass, receipt/hash checks and descriptive report.
If the source lacks dominated HEAD cases, do not fabricate a clean-return fusion
result from this cohort. The next required evidence is a separately scoped
homogeneous-surface control or actual mixed-surface sensor measurements. Existing
RGB results and both failed primary source gates remain unchanged.
