# Near-field spatial support probe V1

Phase: EXPLORE. Scope: depth postprocessing and representation only; no route
selection, autonomous avoidance, Android promotion or real-user claim.

## Question and hypothesis

Under a limited compute budget, can near-field evidence preserve unknown-shape,
thin, low and overhead surfaces for timely directional alerts?

The first testable hypothesis is that stride sampling and region-wide quantiles
discard geometrically visible small surfaces. Three-pixel collinear depth support
followed by tile minima may preserve them while rejecting isolated depth noise.
This is a component hypothesis, not a novel-method or single-camera reliability
claim. Correlated depth artifacts are an explicit counterexample opportunity.

## Fixed comparison

- Input is the same forward-metric depth array and calibrated height/pitch for
  every arm. Attention range is 3 m; gravity-relative height edges are
  0.065/0.65/1.40/1.85 m; camera-heading angular cuts are -10/+10 degrees.
- `STRIDE4_QUANTILE`: the legacy stride=4, offset=2, >=8 samples and 4% quantile
  mechanics adapted to the same nine direction/height cells. This is not the
  unchanged navigation controller or its closed-loop success score.
- `DENSE_QUANTILE`: same adapter at every pixel, isolating sampling from pooling.
- `SUPPORTED_TILES`: three contiguous horizontal OR vertical compatible depth
  samples (range <=0.08+0.02*center_depth m); retain 16x16 tile minima separately
  by height plus exact nine region minima. Weak near evidence is UNKNOWN.
- No time smoothing or learned encoder changes are included. One frame is the
  response opportunity; processing time excludes camera/model/audio latency.

## Sources, budget and stop

1. Run the complete deterministic procedural source once (<=120 frames, 320x240).
   Analytic ground/far-wall and inserted surfaces are a synthetic depth-space
   diagnostic. Expected cells come from visible inserted geometry, never model
   output. Enumerate thin-line pixel phases and widths. Include invalid input,
   isolated noise and 2x2/3x3 correlated artifacts as negative controls.
2. Run all eleven existing `willow-eye170-first-person` RGB-D frames without
   changing the frozen sample, trajectories or assets. Native-depth processing is
   an engineering replay. A fixed existing Depth Anything V2 metric Hypersim Small
   (max_depth=20, input_size=518) supplies the separate RGB prediction branch.
   No per-frame native-depth scale fitting or outcome-based frame selection.
   Simulator height/pitch remain privileged calibration in BOTH branches.
3. Compare CPU/GPU on equivalent representation work through research_backend;
   include input and output transfers. Report actual selected device, per-arm
   P50/P95 timings and numeric output bytes. Model inference is measured separately
   on an available GPU. These are workstation timings, not wearable energy,
   phone latency, process memory savings or neural-network cost reduction.
4. Stop after this fixed comparison and focused correctness checks. No parameter
   search, training, UE launch, protected FIT_ONLY/FINAL access or cohort rerun.
   Mechanical errors may be repaired with the failed log preserved; repairs do
   not permit tuning the method to observed scores.

## Evaluation and decision

Report TP/FP/FN and event-free single-frame opportunity recall per source family;
all nine cells contribute to each frame's denominator. Report false-positive
frames and UNKNOWN separately; unknown does not remove a truth-positive cell
from the recall denominator. On UE replay, compare predicted-depth decisions
with the SAME arm on native depth, calling it agreement with a privileged
reference, not independently labeled obstacle accuracy or a method ranking.

Keep the mechanism as a Development component if it recovers otherwise lost
small-surface evidence at an explicit measured cost. An added correlated-artifact
false alert rules out claiming equal-error improvement without further evidence.
If dense quantiles recover the same evidence more cheaply, prefer that simpler
mechanism. If gains disappear on predicted depth, the depth frontend remains the
next bottleneck. No gain is a scoped negative; missing runtime is NOT_EVALUABLE.

## Completed results

One fixed run completed; no method or threshold was changed after scoring.
Receipts are under `artifacts.local/nearfield/representation-20260907-v1/`.

### Procedural depth-space diagnostic

There are 88 frames and 240 positive direction/height cells, not 240 independent
objects. The source is constructed to probe sampling phase and small geometry.

| Arm | TP / FP / FN | Positive-cell recall | P50 / P95 processing ms |
| --- | --- | --- | --- |
| STRIDE4_QUANTILE (CPU) | 162 / 0 / 78 | 67.5% | 0.620 / 1.756 |
| DENSE_QUANTILE (CPU) | 210 / 0 / 30 | 87.5% | 3.070 / 7.346 |
| SUPPORTED_TILES (CUDA) | 240 / 1 / 0 | 100% | 3.296 / 8.060 |

| Family | Positive cells | Stride-4 TP | Dense TP | Supported TP |
| --- | --- | --- | --- | --- |
| Thin pole | 96 | 60 | 96 | 96 |
| Overhead bar | 96 | 60 | 72 | 96 |
| Near wall | 24 | 18 | 18 | 24 |
| Low block | 6 | 6 | 6 | 6 |
| Irregular L | 18 | 18 | 18 | 18 |

The supported arm rejected isolated noise and the 2x2 artifact but alerted on
the 3x3 artifact. That patch is indistinguishable from a small physical surface
using its single-frame depth alone. No same-error-budget superiority is claimed.
The two all-invalid controls remained UNKNOWN; their absence of alerts is an
interface check, not evidence of free space. No overall accuracy/specificity is
reported from negative or unobservable cells.

The prototype scans the full depth raster and emits 11,136 bytes of numeric tile
and region evidence from 307,200 input bytes (27.59x smaller retained payload).
It does not save depth-model inference or eliminate dense temporary buffers;
the simple baselines emit only 36 distance bytes. This is not process-memory
compression or an end-to-end compute reduction versus the legacy baseline.

GPU/CPU equivalent-encoder probes selected CUDA: at 320x240 CPU/GPU median was
8.789/3.316 ms; at 640x360 it was 22.017/3.158 ms, including transfers and summaries.
The quantile comparators are NumPy/CPU implementations. Cross-arm timings are
implementation costs on this workstation, not an equal-backend speedup claim.
No optimized GPU-quantile or Android performance comparison was performed.

### Fixed RGB prediction branch

All eleven frames of the existing first-person Willow sample were used. The
unchanged metric Hypersim Small model used input_size=518, max_depth=20 m and
CUDA:0 (RTX 5060 Laptop GPU). Weight SHA-256:
`b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545`.

| RGB arm vs its own native-depth reference | TP / FP / FN / TN |
| --- | --- |
| Stride-4 | 0 / 0 / 25 / 74 |
| Dense | 0 / 0 / 26 / 73 |
| Supported | 0 / 0 / 26 / 73 |

All three RGB arms emitted no alerts. Their reference denominators differ and
cannot rank obstacle accuracy. The eleven sequential synthetic frames are not
independent scenes or real-camera validation. Simulator calibration is still
privileged, and an indoor-trained model on this outdoor sample is a specific
model/domain pairing, not evidence against all monocular models.

Inference P50/P95 was 86.166/94.928 ms. Supported-arm RGB postprocessing was
3.911/7.080 ms; paired inference-plus-postprocessing was 89.729/100.942 ms.
RGB decode alone was 4.711/19.416 ms and is excluded from those totals, as are
capture, network, feedback, loading and warmup. Model weights are 99,222,290 bytes;
the observed PyTorch inference allocation peak was 484,094,464 bytes, including
the resident model and other process tensors. No sustained phone/energy claim.

### Saved-prediction loss diagnosis

An outcome-directed read-only diagnostic partitioned 119,721 native eligible
near-surface reference pixels, without rerunning inference or fitting scale:

| Disjoint first loss | Pixels |
| --- | --- |
| Raw predicted depth outside valid range | 2 |
| Reconstructed forward distance >3 m | 67,956 |
| Still within 3 m but reconstructed height <0.065 m | 51,763 |
| Eligible after geometry filters | 0 |

Of the over-distance pixels, 66,379 were also below the height threshold. On
these reference rays, median predicted/native depth ratio was 1.21714, median
forward bias +0.53294 m and median height bias -0.33962 m. All losses precede
spatial support/compression. The reference is overwhelmingly low-height:
118,410 low, 1,305 body and six head pixels, so it cannot establish balanced
coverage of pole/overhead performance. The fixed-scale geometry conversion
amplifies this model's depth bias into both range and height errors.

The diagnostic validated 34 receipt-bound source/observation files and unchanged
post-analysis hashes. It is a synthetic geometry-reference diagnosis, not true
obstacle labeling. No threshold or per-frame scale rescue was attempted.

## Decision and next capability check

Retain spatial support before aggregation as a Development component candidate:
it adds evidence beyond dense quantiles in the procedural source. Keep the 3x3
artifact as its negative control. It is not a complete alert algorithm and is
not promoted to the App or motion policy.

The next useful hypothesis concerns **local ground-relative geometry and depth
reliability**: determine whether an observable local-ground estimate can retain
near-obstacle height evidence despite metric bias, and whether the RGB frontend
actually retains curb/thin-surface boundaries. A local-ground estimate is not a
guaranteed fix; blurred-away geometry may remain absent. This probe does not run
that successor. Further compression/training comes after this failure is resolved.

Registration through `tools/knowledge.py register-experiment` failed before
mutation on the pre-existing `experiments/index.jsonl:252` input-fingerprint
mismatch. `registration.log` preserves the error. Desired component disposition
is `COMPONENT_OR_CHALLENGER / COMPONENT`; structured publication remains pending,
and no new registered terminal or restored evidence freshness is claimed.

## Reproduction and evidence

Use the existing CUDA Python environment with NumPy, Torch, OpenCV and the
already-downloaded metric model source. Choose a fresh output directory.

```powershell
python -m unittest discover -s research/active/dtr-r0/nearfield -p test_near_field.py -v
python research/active/dtr-r0/nearfield/run_probe.py --output artifacts.local/nearfield/representation-new-run --model-dir artifacts.local/unreal/willow-eye170-first-person/model --metric-source artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main/metric_depth --weights artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth
python research/active/dtr-r0/nearfield/diagnose_depth_loss.py --run artifacts.local/nearfield/representation-new-run --output artifacts.local/nearfield/representation-new-run/loss-diagnosis.json
```

Five focused tests passed: invalid observations, retained thin surface and input
immutability, isolated weak evidence, angular boundary preservation and rejected
bad calibration/shape. CPU/GPU probe outputs also matched within 1e-5.

Durable receipts in the original run directory:

- `result.json`: complete code/source/model identities, paired outcomes and costs.
  SHA-256 `e23186a10e6f5021017c7336b8c9490617e6b86ebf7ce27b5225cec673f343c1`.
- `procedural.json`, `procedural-inputs.json`: per-case outputs and analytic input
  hashes/expected cells; `backend-320x240.json`, `backend-640x360.json`: placement.
- `rgb-replay/predicted_depth/`: all eleven saved predictions;
  `rgb-replay/rgb_depth_first_last.png`: fixed-scale first/last visual inspection.
- `loss-diagnosis.json`: disjoint pixel losses, overlap table and verified hashes.
  SHA-256 `48ea55c9272e59344027589562628e0eb954c24fda47c7fb7d5ae826ccef9932`.
- `execution.log`, `registration.log`: actual execution and central publication gap.

Only short task-owned Python processes were launched; the fixed replay completed
and released its model/encoder objects. No simulator, listener or worker lease
was started. Numerical outputs stay under the canonical artifact junction.
