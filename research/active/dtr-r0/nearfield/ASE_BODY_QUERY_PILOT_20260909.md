# ASE BODY/HEAD small-subset pilot

Phase: EXPLORE. Status: `PREPARED_AWAITING_OFFICIAL_DOWNLOAD_MANIFEST`.
This is a preparation record, not a completed ASE experiment or a route terminal.

## Question and bounded decision

Does a small external egocentric source contain useful near-field BODY-only and
HEAD-only visible surfaces under the existing query volumes? The hypothesis is
that ASE supplies geometry variation missing from same-world UE counterfactuals.
The retained model is frozen BODY-QUERY B; the previous attribution recipe R1
remains a negative control. No old source, selection, weights or budget changes.

Use official TRAIN chunk 0, scenes **0, 4, 8**, with **24 uniformly spaced frames
per scene** chosen without viewing model outputs. These are separate synthetic
layouts, not verified asset-disjoint or real-world environments. Maximum one
chunk / 4 GiB compressed and 4 GiB selected extraction. Stop on missing official
access, checksum/schema mismatch or cap exceedance; retain incomplete receipts.
Do not silently switch to a mirror, another source or cherry-picked frames.

Count BODY-only, HEAD-only, both, neither detected and invalid pixels. Inspect
first/middle/last frame overlays for each scene, plus count distributions.
At least 8 HEAD-only frames spanning 2 scenes and 8 BODY-only frames admits a
follow-up model/calibration check; it does not establish method success. Sparse
coverage rejects this subset for the intended HEAD-only comparison, not ASE as
a whole. No training is authorized by a positive coverage result alone.

## Geometry and what it means

- `ase_camera.py` implements the official fixed ASE Fisheye624 calibration for
  704 or 1408 square, unrotated images. It includes radial and tangential terms,
  half-pixel-aware rescaling, the official device-camera transform and a
  conservative 1-radian cone. Invalid projection rays remain NaN.
- ASE depths are Euclidean unit-ray distances in millimeters, not axial Z depth.
  Scale them to meters before multiplying normalized camera rays.
- The trajectory is world-from-device. Compose with device-from-camera; do not
  invert that transform because of the misleading C++ local variable name.
- Use world -Z gravity, horizontal optical heading and an explicitly **assumed
  camera-anchored 1.70 m proxy floor**. This does not measure the user's floor,
  torso orientation or actual camera-to-body transform. Z-up alignment still
  needs visual inspection on actual ASE data.
- Queries preserve `contact_retina_spec.py`: BODY forward [.18,3.18], lateral
  [-.28,.28], height [.65,1.40]; HEAD [.13,3.13], [-.18,.18], [1.40,1.85] m.
  Each is split into 2 distance by 3 lateral cells. Height 1.40 belongs to both.
- Count classes are 0/1/2/3+; >=3 visible pixels per head sets the near bit.
  These are **native source counts**, not the V1 640x360/HFOV100 count contract.
  Invalid pixels and missing visibility are not free-space evidence.

The current B inference module assumes a fixed pinhole camera; simply resizing
ASE would violate that contract. A model comparison additionally requires a
reviewed pinhole resampling and coverage policy, with missing rays exposed.
The current pilot intentionally stops at source geometry before that decision.

## Source evidence

Inspected official sources on 2026-09-09:

- [ASE data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format)
- [Official download workflow](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_download_dataset)
- [Calibration constants](https://github.com/facebookresearch/projectaria_tools/blob/main/projects/AriaSyntheticEnvironment/AseCalibrationProvider.cpp)
- [Fisheye624 formula](https://github.com/facebookresearch/projectaria_tools/blob/main/core/calibration/camera_projections/FisheyeRadTanThinPrism.h)
- [Transform constructor semantics](https://github.com/facebookresearch/projectaria_tools/blob/main/core/calibration/CameraCalibration.cpp)
- [Trajectory reader](https://github.com/facebookresearch/projectaria_tools/blob/main/projects/AriaSyntheticEnvironment/tutorial/code_snippets/readers.py)

The official download metadata is a list with `filename`, `cdn`, `sha` (SHA1).
The local downloader keeps TLS verification enabled, streams bytes, checks SHA1,
records SHA256 and extracts only selected source inputs. Signed URLs and user
contact details must remain outside Git. The license is owned by Project Aria;
public source access does not authorize redistribution of the dataset.

## Reproduction

From `E:/linnan/linnan`, with the provided official URL JSON:

```powershell
$code = 'research/active/dtr-r0/nearfield'
$source = 'artifacts.local/downloads/ase-body-query-pilot-20260909'
$run = 'artifacts.local/evidence/ase-body-query-pilot-20260909/run-01'
& E:/codex-tools/bin/blindassist-research-gpu.cmd -B "$code/ase_body_query_pilot.py" download `
  --cdn-file PATH_TO_OFFICIAL_URL_JSON --output $source
& E:/codex-tools/bin/blindassist-research-gpu.cmd -B "$code/ase_body_query_pilot.py" audit `
  --source "$source/scenes" --output $run
```

Before actual ASE decoding, register the run with
`python tools/knowledge.py register-experiment` using this brief and the verified
download receipt as inputs. Record the result and its inheritance after the
coverage decision; preparation does not create a completed terminal.

Artifacts resolve through the canonical junction to
`F:/ba-data/blindassist-artifacts-20260805/`. Use the existing Python 3.11 research
runtime (NumPy/Pillow/SciPy/requests); no new system environment is required.
The geometry-only tiny checks use CPU and allocate no GPU model or UE process.

## Validation and current limit

Eleven focused geometry/calibration tests passed. They check distance versus Z,
world rotation and gravity invariance, pitch retention, query boundary ownership,
counts and UNKNOWN, unit-ray reconstruction, optical center, image rescaling,
and invalid/out-of-cone rejection. Mathematical round trips are not independent
SDK parity or proof of correct geometry on ASE images.

The complete CLI pipeline also passed on a self-generated front-wall fixture
(3 fixture scene IDs, 2 frames each). All 6 frames yielded the expected BOTH
label. Download transport was mocked; ZIP creation/extraction, official-format
SHA1 metadata checking, SHA256 receipt, image decoding, trajectory composition,
counts, JSON and preview generation ran normally. This fixture is explicitly
not an ASE sample and cannot support ASE coverage claims. Its durable receipt is
`artifacts.local/evidence/ase-body-query-pilot-20260909/engineering-01/engineering-receipt.json`.
The six-frame audit took 6.234 seconds on NumPy CPU, without a model allocation.

Actual download is pending: the official form needs its email/access flow to
produce the URL JSON. Both Edge and in-app browser page-state reads timed out;
opening the application page succeeded but automated form submission did not.
No ASE source pixels, coverage metrics, model outputs or training results have
been produced. Existing route conclusions remain unchanged.
