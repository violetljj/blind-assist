# BlindAssist Last-10m current-frame completion S2-S5 result (2026-08-22)

## Outcome

`FRESH_CURRENT_FRAME_COMPLETION_NOT_ESTABLISHED` on all four sealed cohorts.

The public-data pipeline is established and fully automated. It does not ask the user to collect data, select images,
label frames, run commands, or adjudicate intermediate output. The current bottleneck is not data acquisition and is no
longer primarily proposal availability; it is safe current-frame functional/range selection among available candidates.

No result below supports a real-world building-entrance, traversability, navigation, product, or safety claim.

## Input and leakage boundary

The materializers obtain RGB, depth, and segmentation from the public
[TartanAir dataset](https://huggingface.co/datasets/theairlabcmu/tartanair2). For each cohort they:

1. freeze a public goal contract before model execution;
2. derive a roster by fixed geometry/depth rules rather than observed provider output;
3. expose RGB, aligned public depth, and goal semantics to the provider;
4. retain legal target boxes, target identity, near/far strata, and completion adjudication in private evaluator input;
5. hash public input, private truth, manifest, run, journal, and evaluation artifacts.

S2-S5 use distinct environments. Development training excludes every formal environment it later evaluates. Frozen
formal cohorts are not replayed after their result is observed.

## Sealed results

| Cohort | Environment | Cases | Candidate available | Opportunities | Decisions | Correct | False | Correct coverage | Evaluation SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| S2 | RetroOffice + CountryHouse | 48 | 42 | 11 | 4 | 4 | 0 | 36.36% | `8e09113ee146d583364817c1fd0d1c61d9701866d5d5290a49fd339bb5093c71` |
| S3 | AmericanDiner | 48 | 33 | 5 | 5 | 4 | 1 | 80.00% | `f69a2b5ff971ae988ee99abfa2877174d99185732ab5aec76b5e06dfbcc364fc` |
| S4 | House | 48 | 43 | 24 | 5 | 1 | 4 | 4.17% | `be60b37963d85c9ea656bcb7a6b97c5f30781e852a388bc255e14928e5c35dcd` |
| S5 | Office | 48 | 40 | 24 | 14 | 8 | 6 | 33.33% | `dc6d42a4a457095cb6c9b9fa33ee142e46114012899fea4f28c78238bb6d6065` |

Every terminal is `FRESH_CURRENT_FRAME_COMPLETION_NOT_ESTABLISHED`. S3's apparent 80% coverage cannot override its
single false completion and five-opportunity denominator.

## Failure-layer attribution

S4 supplies the cleanest localization: all 23 missed completion opportunities already contained a YOLOE candidate that
matched the legal target at IoU >= 0.30. The target was therefore observable to the bounded proposal pool, but the fixed
controller did not safely select/commit it. This establishes proposal availability only for this synthetic door subset;
it does not establish object detection generally or instance identity.

The following development-only routes were attempted without consuming a new formal cohort:

- fixed public metric-depth aperture rule: `17/24` correct and `0` false on consumed S4, but only `8/24` correct with
  `6` false on untouched S5 when frozen into the formal provider;
- ADE20K semantic aperture fusion: `5` correct and `2` false on consumed S4, `NOT_PROMISING`;
- public TartanAir door-semantic model trained on OldBrickHouseDay, Hospital, and Restaurant: validation door IoU
  `0.475`, but no completion decisions on consumed S4/S5, `NOT_PROMISING`;
- RGB candidate verifier trained on S2-S4: held-out balanced accuracy `0.632`; S5 development completion `3` correct,
  `3` false, `NOT_PROMISING`;
- fixed 21-feature public depth-structure Random Forest trained on S2-S4: S5 candidate balanced accuracy `0.721`,
  completion `16/23` correct with `5` false, `NOT_PROMISING`.

The S4-to-S5 reversal rejects the tempting conclusion that a single within-box depth percentile reliably denotes an open,
reachable aperture. Another threshold search on the same consumed environments would be post-outcome tuning and is not
authorized.

## Scientific conclusion and next gate

The narrow established facts are:

- public goal semantics, RGB-D observations, automatic roster construction, and private evaluation can be materialized
  without user labor or evaluator-to-provider leakage;
- goal-semantic YOLOE proposals frequently contain the legal target on these synthetic cohorts;
- the tested bbox, semantic, RGB, and hand-crafted RGB-D selection signals do not establish safe completion control.

A fresh successor is not authorized by the current verifier. The next development lane must change representation to
explicit functional free-space/traversable-aperture grounding or temporal geometric evidence, then meet a predeclared
zero-false and minimum-coverage development gate before consuming another untouched cohort. It must not recover private
target identity through prompts and must not turn user labor into an input requirement.
