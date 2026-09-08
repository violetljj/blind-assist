# City 500-group acquisition

The user authorized further throughput optimization followed by actual worker
acquisition of 500 groups. This is controlled Development data in one existing
map, not independent-world validation or model training.

## Acquisition changes

`city_pcg_capture.py` now separates routine `settling_ticks` from
`first_use_settling_ticks`. The collection requests 8 routine ticks and 32
post-readiness ticks for the initial view, a newly used mesh, readiness waits
over 0.5 s, or a camera XY displacement over 10 m. Native readiness remains
mandatory; the slower first-use pass addresses the previously observed missing
RGB asset defect. Existing specs retain their previous tick count by default.
Progress JSON is written at the start of a settling pass and at most once per
second during it, rather than on every render tick. Asynchronous pair exports
retain bounded backpressure and completion checks.

The entire collection runs in one editor lifecycle. The 640x360 RGB, float32
native depth and support masks retain their resolution and formats. It uses
the explicit async/no-appearance profile; the previously documented background
differences from the standard preview profile still apply. No full-scene
pixel/depth equivalence is claimed.

An initial same-150-case comparison lowers the prior production profile from
32 to 8 ticks: complete job 127.286 to 66.234 s, editor lifecycle 115.233 to
54.275 s, and script 93.218 to 31.156 s. Native capture, CUDA verification and
grouping pass. All 150 risk states match; across all frames there are zero
known BODY/HEAD support-value changes and zero support-depth changes over 3 cm.
First-use props and plaza imagery are inspected. This initial comparison used
8 ticks for its first-use guard too; the final 500-group run adds the separate
32-tick guard, so its actual throughput must be measured independently.
Background variation persists and is not claimed pixel equivalent. Evidence:
`artifacts.local/nearfield/city-pcg-20260908/worker-scale500-v1/eight150-evidence/`.

## Dataset design

`tools/make_city_obstacle_suite.py --collection-500-groups` creates:

- two existing map regions, west sidewalk and plaza;
- ten poses per region, with camera heights 1.6/1.7/1.8 m above the floor;
- bicycle, scaffold frame, barricade, picnic table and park bench;
- five front-bound placement distances: 0.9, 1.5, 2.3, 3.5 and 6 m;
- clear, centered object and right-clearance variants per group.

This is 500 groups / 1,500 frames with 500 distinct center camera/object
transforms. It does not provide 500 unique backgrounds: there are 20 camera
poses and one world. Object yaw and camera height vary with pose; this is not
a full factorial design or class-balanced HEAD-risk dataset. Placement bounds
are anchors only; native visible surfaces supply the labels. Identical clear
controls at a pose remain useful paired controls but are not independent views.

The source spec is
`artifacts.local/nearfield/city-pcg-20260908/collection-500-spec-v1.json`, SHA256
`19a90e1dfd43a6bf3f4aeb99f083749fc29ab1505e89ab7e6d772260e9954b98`.
Before dispatch, all 500 triplets pass group validation and all 500 center
transforms are unique. The map and asset paths are verified by the generator.
Worker adaptation records the source hash and remaps only machine-local paths.

Acceptance requires completed native capture, CUDA geometry verification,
group QA and visual inspection of representative output across both regions.
Keep raw output on the worker, return thin hashes/reports/previews, and retain
DDC/Zen while releasing task-owned editor processes and scheduled jobs.

## Completed worker result

`collection500-v1` completes all 1,500 frames and 500 groups. Native capture,
CUDA geometry verification and group QA are PASS. All 500 center transforms
are unique in the actual worker input. There are 1,500 submitted/completed
native exports, zero failed exports and peak pending queue depth one.

| Measurement | Actual result |
| --- | ---: |
| Editor lifecycle | 261.915 s |
| Capture script | 237.203 s |
| CUDA geometry verification | 35.448 s |
| Group summary | 32.038 s |
| Complete worker job | 346.527 s (5 min 47 s) |
| Total output | 2,869,521,426 bytes / 4,527 files |

Canonical RGB occupies 784,217,583 bytes, depth 1,382,592,000 bytes and support
masks 691,392,000 bytes. No reference or appearance payload is generated.
The actual first plaza transition at index 750 records `view_jump=true`, a
1.156 s readiness wait and 32 post-readiness settling ticks. Primary visual
inspection of the 20-pose contact sheet confirms representative props render
in both regions. This is representative visual QA, not manual inspection of
all 1,500 images or evidence of real-world model performance.

Raw data is retained on the worker at
`G:/DevWorkspace/BlindAssist/artifacts/work/city-pcg-20260908/collection500-v1`.
Primary evidence, metadata and previews are under
`artifacts.local/nearfield/city-pcg-20260908/worker-scale500-v1/collection500-evidence/`.
Task-owned editor processes and scheduled jobs are released; DDC/Zen remain.

The measured complete-job rate is 4.33 frames/s, including startup and QA.
At this workload and batch size, linear estimates are 38.5 minutes for 10,000
frames and 1.93 hours for 10,000 three-frame groups. These larger quantities
have not been captured; different assets, render settings, cold starts and
thermal conditions can change the rate. Native encoder time totals 47.01 s
and overlaps capture; geometry/grouping now account for roughly 67.5 s of
the job. Those stages remain enabled rather than being hidden from throughput.
