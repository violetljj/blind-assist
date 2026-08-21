# P0 GoalGrounding-Silver Feasibility Audit

审计日期：2026-08-21
状态：`COMPLETE / DESIGN_ONLY / FEASIBLE_WITH_REQUIRED_CONSTRAINTS / NO_DATA_OR_MODEL_EXECUTION`

## Executive verdict

`Mapillary + Overture Buildings/Places + OSM entrance=* + project-terraforma/mapillary-entrances` 可以成为
BlindAssist P0 的自动构造 real-world silver cohort，但必须把 upstream pipeline 降格为候选生成器和几何
派生器，把 OSM entrance/建筑拓扑、跨源 building crosswalk 与多视角一致性设为独立 admission gates。

“零人工标注”只表示本 cohort 不新增 BlindAssist 专用人工标签；OSM 和其他上游数据仍含社区或供应商的
人工贡献，不能描述为全谱系无人参与。

最终 verdict：`FEASIBLE_WITH_REQUIRED_CONSTRAINTS`。

本次使用 Exa 检索 9 个互补 workstreams 的 85 个 search hits，并将证据收敛到官方文档、许可证、OSM
Foundation 指南与固定 GitHub commit 的源码。没有下载/物化任何 cohort，没有调用模型或运行 baseline。

## 1. Source and interface audit

| Component | Confirmed capability | Material limitation | Authority in P0 |
|---|---|---|---|
| Mapillary API | image/entity and coverage surfaces; image/sequence IDs; original/computed geometry; compass metadata; capture time; camera/pano and thumbnail fields | API token required; bbox/radius constraints; metadata is uncertain; imagery attribution and Terms apply | real RGB + observation metadata |
| Overture Buildings | global conflated building/part geometry, GERS IDs and per-source provenance | footprint may be roofprint or ML-derived; source quality varies | building geometry prior |
| Overture Places | POI ID/name/category/geometry with per-record source/license | Place point is not a proven building or entrance; multi-license records | target identity prior |
| OSM `entrance=*` | entrance node types, normally on building/area outline; `main`, `secondary`, `service`, access/wheelchair/door fields | incomplete, stale or inconsistently mapped; `main` does not mean accessible/public/open | independent map assertion |
| mapillary-entrances | bounded Overture retrieval, Mapillary thumbnails/metadata, YOLO candidates, ray-wall association, snapped entrance point and clustering | no OSM entrance ingestion, no independent truth, no implemented multiview triangulation | candidate/mining scaffold only |

Primary references: [Mapillary API](https://www.mapillary.com/developer/api-documentation),
[Mapillary Terms](https://www.mapillary.com/terms),
[Overture Buildings](https://docs.overturemaps.org/guides/buildings/),
[Overture Places](https://docs.overturemaps.org/guides/places/),
[Overture source fields](https://docs.overturemaps.org/schema/reference/common/source_item/),
[OSM entrance key](https://wiki.openstreetmap.org/wiki/Key:entrance).

## 2. `mapillary-entrances` source audit

Audited repository: [project-terraforma/mapillary-entrances](https://github.com/project-terraforma/mapillary-entrances),
commit `3d3b85244b1a1ec2ba05a997d56d000936cc554a`.

### What it actually provides

- [`imagery.py`](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/src/imagery.py)
  requests `id`, `computed_geometry`, `captured_at`, `compass_angle`, `thumb_1024_url`, and `camera_type`, downloads
  thumbnails and records coordinates/heading. It does not retain sequence ID in its emitted record.
- [`download.py`](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/src/download.py)
  retrieves Overture building `id/geometry/bbox` and Place `id/geometry/names/categories/bbox` into local Parquet.
- [`duck_db_utils.py`](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/src/utils/duck_db_utils.py)
  joins Places to Buildings by containment or proximity; it computes centroid distance.
- [`matching_utils.py`](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/src/utils/matching_utils.py)
  chooses one Place using inside/overlap/distance/category heuristics. This is an association prediction, not source truth.
- [`inference_utils.py`](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/src/utils/inference_utils.py)
  runs a YOLO checkpoint, uses bbox bottom-center plus assumed FOV and compass heading to cast a local ray, intersects
  building walls, chooses an exterior segment and snaps the hit near the wall.
- [`inference.py`](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/src/inference.py)
  emits `bid`, entrance coordinate, image path, hit and wall segment, then clusters same-building predictions within 5 m.

### Important corrections to the initial proposal

1. The README says “geometry-based triangulation,” but the audited implementation is per-image 2-D ray/segment
   intersection plus cross-image proximity clustering; it does not solve a multiview triangulation system.
2. The code uses `computed_geometry` with `compass_angle`, assumes a 45° non-pano FOV, and does not preserve explicit
   pose uncertainty. A future protocol implementation should request and prefer supported computed heading/FOV fields,
   retain raw and computed values, and fail closed when calibration is absent.
3. The line `find("entrance") != -1 or True` keeps every detected class. With a single-class checkpoint this may be
   harmless, but it is not a valid class filter for arbitrary weights.
4. The model is loaded inside the per-image validation function, which is an efficiency issue but not evidence.
5. The upstream output lacks OSM entrance IDs/types, sequence ancestry, per-source Overture licenses, detector scores in
   final records, independent semantic evidence, uncertainty propagation and evaluator isolation.

Therefore the upstream project is reusable only as an Apache-2.0 engineering scaffold after a pinned-source audit. Its
predictions must never be imported as P0 truth.

## 3. License and policy audit

This section records design requirements, not legal advice. Distribution plans require project-owner/legal review against
the then-current terms before materialization.

### Mapillary

The [current Terms](https://www.mapillary.com/terms) state that other users' User Content is generally CC BY-SA unless
otherwise indicated; individual images served from local infrastructure require visible Mapillary attribution/linking;
data extracted through the API/vector tiles also requires visible source attribution. The Terms permit commercial
improvement/training/development uses, but prohibit using Mapillary Services with real-time navigation or route guidance.
This P0 offline benchmark design is not a real-time guidance integration; a future product path must be separately reviewed.
The [Mapillary license help](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)
also describes imagery as CC BY-SA and gives attribution guidance.

Frozen requirements: retain image ID, creator/organization where supplied, image-page URL, license, attribution string,
capture date and retrieval receipt; do not unblur/re-identify; do not redistribute an image bundle until CC BY-SA and
Mapillary Terms obligations are satisfied.

### Overture

[Buildings](https://docs.overturemaps.org/guides/buildings/) are published under ODbL because the conflated theme includes
OSM. [Places](https://docs.overturemaps.org/guides/places/) are multi-license: CDLA Permissive 2.0, Apache-2.0 and CC0
depending on source. The [attribution page](https://docs.overturemaps.org/attribution/) and each feature's `sources[]`
must be retained; Foursquare-sourced Places have Apache/NOTICE/change-notice duties. Do not stamp the entire Place theme
as one license.

### OpenStreetMap

OSM data is ODbL and requires attribution/share-alike for distributed OSM or derivative databases; see the
[OSM copyright page](https://www.openstreetmap.org/copyright/en) and OSM Foundation guidance. Joining OSM anchors to
Overture/Mapillary-derived records may form a derivative database depending on structure and public use. The safest frozen
design is to keep source-native tables and licenses separable, retain the reproducible join recipe and IDs, and require an
ODbL classification decision before releasing the joined benchmark database.

### Upstream code/model

Repository source is [Apache-2.0](https://raw.githubusercontent.com/project-terraforma/mapillary-entrances/3d3b85244b1a1ec2ba05a997d56d000936cc554a/LICENSE).
That license does not automatically cover Mapillary imagery, Overture/OSM data, downloaded Hugging Face weights or every
dependency. Each artifact needs its own license/NOTICE ledger.

## 4. Scientific validity audit

### Why the route is viable

- Real RGB avoids making a synthetic visual domain the primary benchmark.
- OSM entrance/building topology supplies a non-vision map assertion.
- Overture supplies broad building geometry and named Place identity with source provenance.
- Candidate-to-wall geometry makes wrong-building errors observable and creates structured hard negatives.
- Mapillary sequences make cross-view consistency possible once explicitly implemented.
- Fail-closed admission can trade coverage for label precision without human patching.

### Why constraints are mandatory

- OSM coverage is missing-not-at-random and cannot define a representative population denominator.
- Map/pose/footprint errors are correlated with geography, capture device and urban form.
- Multiple teachers may share training data or architecture, so vote count is not independence.
- Mapillary sequence frames are repeated measures, not independent examples.
- Without human audit there is no empirical gold-label error rate; high agreement is an internal consistency property.
- A teacher-overlapping evaluated system can overfit the silver construction process.

### Admission consequence

Only `SILVER_A_PRIMARY`—unique map anchor + verified geometry + verified multiview + no conflict + no evaluator lineage
overlap—enters the primary scientific subset. Map/geometry-only cases are secondary; teacher-only cases are development;
conflicts are rejected. VLM confidence never repairs a failed independent gate.

## 5. Schema and circularity review

The companion schemas require source releases/records/licenses/hashes, goal/POI, target and visible buildings, image/pose
metadata, candidate bboxes, `entrance_of` relations, target visibility authority, hard negatives, evidence payloads,
quality gates, abstain reasons, difficulty strata and split ancestry.

Circularity is blocked by lineage groups, candidate-source recording, hidden evaluator truth, mandatory overlap reporting
and primary-subset exclusion when the evaluated system shares teacher lineage. A single model's candidate and relation
judgement count as one lineage.

Exact OSM entrance coordinates, target-building crosswalk and visibility/label fields remain mining/evaluator-only. A
candidate system may receive only the separately frozen coarse POI prior; exposing answer coordinates would turn the task
into map lookup leakage rather than visual goal grounding.

## 6. Supported and unsupported claims

Supported after separately authorized, correctly frozen execution:

- performance on the admitted map-anchored real-world silver subset;
- candidate coverage and goal-conditioned entrance ranking;
- wrong-building/wrong-entrance error and geometry-proven abstention;
- failure anatomy by declared strata and coverage.

Unsupported:

- gold-standard accuracy or measured label correctness;
- accuracy outside the admitted/mapped/Mapillary-covered population;
- human navigation benefit, completion, accessibility, public/open entrance status or safety;
- production/default-App readiness;
- superiority on independent human truth.

## 7. Minimum next slice

If new authorization is granted, begin with a one-area, metadata-only, zero-model mechanics canary. Pin source snapshots,
build only POI-building and OSM entrance-building crosswalks, validate CRS/threshold sensitivity/license ledger/schema and
emit a rejection ledger. Do not download RGB or call a detector/teacher in that slice. Stop for review before any cohort
or model execution.
