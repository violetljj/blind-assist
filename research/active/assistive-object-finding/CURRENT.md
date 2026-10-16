# Assistive object finding

所属：**毕设线**，不是避障线（见[两条工作线](../../../docs/WORK_LINES.md)）。

2026-10-15: Development feasibility pilot completed. Keep object finding as a
candidate thesis direction; do not choose the topic from this pilot alone.

[Results and reproduction](README.md) · [Plan and budget](PLAN.md) · [Run record](RUNS.md)

User-authorized [full data census](functional_parts/FULL_DATA_CENSUS.md) is now
complete for IIT native train/val, PACO annotations and current InstructPart release.
[Training queue](functional_parts/FULL_BASELINE_PROGRESS.md) is RUNNING on the
registered secondary worker after the user reported primary-host memory pressure.
The primary GPU queues were intentionally stopped with the epoch17 checkpoint
preserved; do not auto-restart them. Worker ownership is recorded in the payload's
`migration/STATE.json`. SegFormer and DINO each completed80epoch and full-val
scoring/galleries on train4417/cleanval1757. PLSP40epoch and full Seen test are
COMPLETE; UMD transfer scoring/final summary are still running. An observed
native-void(-1) reader failure was repaired against author code, with original
predictions and failed records preserved; see the training queue record.
The [full point-gap diagnosis](functional_parts/DINO_POINT_GAP_ANALYSIS.md) reproduced
all3821 queries: stable decoding shrinks DINO's point deficit from73 to21,
and predicted-region interior points reach3384 hits. Natural-image small-part
three-candidate oracle remains209/392, so fine-detail/new-candidate generation
with point-quality learning is a hypothesis; no new structure is trained yet.
The repaired post-queue continues UMD evaluations; no thesis topic is selected.
Native labels do not establish free-task role truth; InstructPart release is
incomplete/overlapping. Wait for PLSP and cross-source errors before choosing a thesis.

Public data and the automatic pipeline work: 12 source videos, 16 query sequences,
528 query-frames. DINO localized 82/327 visible frames; CLIP reranking 70/327,
source-grouped comparative learning 81/327. A disclosed posthoc full-pool CLIP
probe reached 85/327 but saved 69 and harmed 66 frames, not a stable improvement.
At 119 localization calls, adaptive KLT gained 8 visible frames over the matched
uniform replay but added 19 absent-target outputs. Neither hypothesis has a
reliable gain in its current implementation.

Official StreamingRVOS-1B cached predictions were rescored, not executed locally:
147/327 visible frames localized, 88/105 absent frames still nonempty. Independent
mask decoding and mapping checks agreed. Its missing 96 no-target rows remain
NOT_EVALUABLE. This GT-enriched MeViS subset is a cross-domain diagnostic, not a
BLV user error rate, full benchmark result or first-person finding validation.

Ego4D access request was submitted through the official portal. The user reports
the agreement signed; official approval/AWS credentials remain pending, and no
local download credentials were found. RefEgo annotations parsed, pixels NOT_RUN.
The next useful experiment is a bounded first-person RefEgo sample after access,
prioritizing retention of low-score correct candidates and specified-instance
presence. Decide whether temporal allocation deserves a thesis chapter afterward.
No further model/data/parameter expansion in this completed pilot.

Payload: `artifacts.local/work/assistive-object-finding-pilot-20261015/`.

An independent user-requested [alternative screen](audio_screen/README.md) is
complete: public audio inference works, but the richer gate showed a tradeoff.
The subsequent [functional-part pilot](functional_parts/README.md) ran released
SelectiveCL on 280 images (AGD 80cal/160eval; IIT 16cal/24eval). Peak-guided crop
did not beat equal-call flip fusion: AGD point proxy .3697 versus .3769; IIT
4/24 versus 4/24. Small GT regions are harder, but crop refinement barely helps.
Confidence filtering helps same-domain point quality while retaining many
cross-domain errors. The authorized [multi-candidate successor](functional_parts/MULTICANDIDATE_RESULTS.md)
also completed on the reused280 images: combined local verification reached
AGD .3449/IIT4 of24, versus equal-call TTA .3534/5 and fixed crops .3218/7.
Candidate AGD oracle .5300 exposes a selection gap, but actual verification
saved19/harmed37 versus first-candidate pointing; matched-coverage rejection
did not establish a stable advantage.

The [learned-ranking follow-up](functional_parts/LEARNED_RANK_RESULTS.md) is now
complete: fit158/cal79, fresh200 project-unseen images after duplicate screening.
Ranking reached .361902 versus matched MSE .352627 and ridge .364588;
paired gain .009275 has 95% interval [-.014708,.033532]. All three individual
ranking seeds lost to their matched MSE seeds. Candidate oracle has144/200
heatmap hits but ranking selects76; rejection has no stable advantage either.
Do not tune on that fresh set or select a thesis from that result.

The [dense visual follow-up](functional_parts/VISUAL_RANK_RESULTS.md) now uses the
same fit158/cal79 and another fresh150. DINO point MLP reaches .492157/84 hits,
versus matched GAP .422118/66 and score-only .433490/70; point-GAP delta .070039,
95% interval [.019425,.120224], all3 seed directions positive. Candidate fusion
.456105 and fixed-crop DINO tree .467660 remain statistically unresolved controls.
Oracle123 hits versus selected84 still leaves39 missed opportunities.

The [cross-source native-mask test](functional_parts/CROSS_SOURCE_RESULTS.md) is
complete: another150 IIT images, frozen AGD models/gates, no target-domain training.
Local DINO hits44/150 versus GAP48, scores54, fixed DINO tree42, fusion49 and
original47. All3 local seeds lose to matched GAP/scores. Native-empty20 remain
in denominator; candidate oracle81, selected44 leaves37 missed opportunities.
Inherited gate retains79 with46 wrong native labels. Same-source positive signal
has not transferred to IIT grasp5/wrap9 regions. Do not select a thesis, claim
general functional correctness or tune this cohort. Supervision alignment and
another held-out source/scene remain future tests. Reports preserve full evidence.
