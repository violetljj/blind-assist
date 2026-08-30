# DTR-CARLA X24 plan-adherent obstacle-risk Development result — 2026-08-30

## Decision

`DTR_CARLA_X24_PLAN_ADHERENT_DEVELOPMENT_GATE_MET`

X24 is retained as the C2 Development successor candidate. On the final 1280×720 CARLA C2 source, it turns the CONTACT episode's eligible complete-occlusion interval from `0 / 8` route-risk frames under X23 into `8 / 8`, a gain of `+100 percentage points`. It also moves the first CONTACT alert from `1.05 s` to `0.90 s`, increasing lead before the `3.85 s` contact from `2.80 s` to `2.95 s`. The SAFE twin has `0` new X24-only false-alert segments after the predeclared fork-plus-grace boundary.

This is a same-source scripted-CARLA **Development** result. It is not blind, source-disjoint, real-world, or product-safety confirmation.

## Result

| Frozen metric | X23 observed-CV route | X24 issued-plan adherence | X24 change |
|---|---:|---:|---:|
| CONTACT eligible full-occlusion coverage, samples `22–29` | `0 / 8` (`0%`) | `8 / 8` (`100%`) | `+100 pp` |
| CONTACT first alert time | `1.05 s` | `0.90 s` | `0.15 s` earlier |
| CONTACT lead before contact at `3.85 s` | `2.80 s` | `2.95 s` | `+0.15 s` |
| SAFE post-fork-plus-`0.70 s` new X24-only false-alert segments | — | `0` | gate met |
| NO_PLAN route-risk identity on `ep_03` and `ep_04` | reference | `62 / 62` frames equal | fallback met |

The CONTACT and SAFE pair has the same observation prefix and the same `0.40 s` complete physical occlusion at samples `22–29`. X24 also produces route risk during that identical SAFE prefix; those frames are deliberately not counted as false alerts because the outcome has not yet causally diverged. The frozen false-alert score begins at `2.70 s`, after the `2.00 s` fork plus `0.70 s` grace, and finds zero new X24-only segments.

All eleven predeclared score checks passed. The result decision is `RETAIN_X24_AS_C2_DEVELOPMENT_SUCCESSOR_CANDIDATE`.

## Structural upgrade

X24 is not a threshold sweep over X23. It changes the online route representation while keeping a conservative fallback:

1. A YOLO11n-seg detector is applied to the truth-blind wearable RGB stream; masks are projected through aligned metric depth into a fixed `160×90` angular-depth grid and then into the CARLA world frame.
2. A shared causal tracker estimates obstacle motion only from current and past model-visible observations. A bounded `0.60 s` HOLD preserves already-observed evidence across temporary visual loss.
3. X23 evaluates collision risk against the wearer's observed-current-velocity route.
4. X24 instead evaluates the immutable issued time-parameterized plan when online adherence is plausible: position residual at most `0.45 m` and velocity-direction error at most `25°`.
5. Missing, invalid, or non-adherent plans fall back to X23. The two NO_PLAN layouts demonstrate exact framewise identity on all `62` frames.

The frozen route horizon is `3.0 s`, route tube radius `0.65 m`, minimum closing speed `0.05 m/s`, and track confirmation time `0.10 s`. The detector ran at confidence `0.10`, NMS IoU `0.50`, and maximum `100` detections, producing `2,763` metric candidates over `224` frames. Predictor input excludes actor state, evaluator truth, outcome, contact, future trajectory, instance visibility, and witness imagery.

## High-resolution C2 source

The scored source is the final synchronized C2 root, not the older 320×180 C1 engineering canary.

- `3` layouts and `4` episodes;
- `74` unique actual CARLA blueprint types;
- wearable RGB, metric depth, instance segmentation, and witness RGB all at `1280×720`;
- `224` frames per sensor and `896` raw sensor payloads;
- deterministic RGB-D replay/alignment receipt with a verified depth-minus-wearable source-frame offset of `70` in every episode;
- CONTACT/SAFE counterfactual pair plus two distinct NO_PLAN layouts.

The old C1 canary remains useful only for engineering smoke checks and is not included in this metric claim.

## Frozen evidence

- C2 evidence root: `E:\linnan\CARLA\experiments\dtr-carla-c2-rich-scene\evidence\c2-rich-sync-20260830-030054`
- X24 result root: `E:\linnan\linnan\artifacts.local\evidence\dtr-carla-x24-plan\c2-rich-sync-20260830-030054`
- Sealed model-package manifest SHA-256: `F5E61479E5586018A866BFDA1B58A2E52D77ED20D759FF2E9C145D231C8092D5`
- Consumed `model/manifest.json` SHA-256: `E482D6D1D1C8A070A922478FEB01F32148263718A5F0F2F9422B6FF561C71B17`
- Sealed full-evidence manifest SHA-256: `B3DD5961C4002FC2D8F4128B92AF83C21DA0B0329220A5B28247C48B89430C6E`
- RGB-D alignment receipt: `A6F0F5D8672A98DDF3B923BE19F7838546F87B8850A3253FEE6C168BC390EED5`
- Detector checkpoint SHA-256: `55ED65C56C91713D23E8402371C6C49A6FD84F257F7DCE452E8D70E41DCBE152`
- Truth-blind freeze SHA-256: `A93B6100FB0C396B9CE8929443C24CDEAAE590D18EF1CC8E42A87B4E10B644B4`
- Prediction SHA-256: `F531918D4A2D330C77027A5147DD28206515BD0BADA6D803DCCEB3A796B434F7`
- One-shot score result SHA-256: `1D92B5031458B70EFD4A419E361805D13E4600DB49E3895405858B942C22F89F`
- Vector result board SHA-256: `BA97D782DFF8212FAA1E734C477F880DA5F828D9D1E14367FE83E8F4C9217D37`

`F5E6…` is the hash of the outer sealed model-package inventory. `E482…` is the hash of the model root manifest actually consumed by the predictor and is recorded inside both sealed inventories; the two hashes identify different layers of the same frozen source rather than different runs.

The freeze was written before prediction, and the one-shot score-attempt receipt was consumed before evaluator truth was opened. The scorer verified the prediction/freeze chain, sealed evidence manifest, and every evaluator file used for scoring.

## Gate checks

- X24 CONTACT first alert exists and has at least `0.40 s` lead;
- X24 covers the complete eligible `0.40 s` CONTACT occlusion window;
- X24 has positive CONTACT benefit over X23;
- SAFE twin has zero new X24-only false-alert segments in the scored tail;
- both NO_PLAN episodes use observed-CV fallback and match X23 framewise;
- all `727` HOLD states stay within the declared `0.60 s` evidence-age limit, with zero violations;
- the single score attempt and all frozen hashes verify.

## Claim boundary and next evidence

This result proves that issued-plan-aware route reasoning plus online adherence and bounded causal track memory can materially outperform the observed-current-velocity route on this controlled C2 occlusion cohort without creating a post-divergence SAFE false-alert segment. It does not prove general CARLA robustness, source-disjoint generalization, sensor-domain transfer, or real-world safety.

The next promotion boundary is a frozen source-disjoint confirmation cohort that preserves the X24 implementation and constants. The Development cohort must not be retuned and then relabeled as confirmation.
