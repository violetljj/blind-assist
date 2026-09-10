# MZ8: one single-frame echo attribution probe

EXPLORE, 2026-09-10. MZ7 showed all49 thin far opportunities both-negative,
not correct-branch suppression. Test whether a frozen position-preserving visual
map can score observed echo hypotheses before body-query aggregation. MZ5 remains
the baseline, temporal stays off. No hardware or safety interpretation.

First recache unchanged frozen64-channel18x32 RGB maps, verify their original
query-pooled features against MZ1/MZ6 caches, and describe paired thin-target map
responses. This identifies available response, not semantic recoverability.

One small shared 65->32->1 MLP receives local64-channel RGB feature and range/4.
Each8x8 zone retains its angular footprint, sampled at7x7 subcell midpoints, and
both observed ranges. Hypothetical ray/range points are eligible only inside the
four fixed body query boxes. Learned evidence, not intersection alone, is pooled
by masked maximum; four query biases may adjust readout. No-return stays UNKNOWN.
This is finite quadrature over possible attribution, not true return ownership.

One seeded fit (108), 1200 Adam steps, lr0.001, batch16, eight old TRAIN_ONLY
and eight MZ6 development frames per step. Frozen backbone. Old2500 TRAIN frames
plus all200 consumed MZ6 frames supply training; these200 are explicitly targeted
training/regression, never confirmation. Native depth supplies auxiliary candidate
compatibility labels: sampled valid radial native distance within0.10m of the
observed echo. Invalid depth is ignored. Query BCE plus0.25 balanced positive/
negative candidate BCE; native, target identity, exact pose and paired empty RGB
are never inference inputs. Old1000 DEV supplies only four thresholds maximizing
TP with per-query FP no greater than MZ5; ties prefer fewerFP then higher threshold.
No old1500 EVAL scoring or new threshold sweep on MZ6. CUDA extraction/training.

Same weights/thresholds, compare correct correspondence and RGB tokens shifted
four zone columns cyclically, geometry and packets fixed. Report clean and MZ6
fixed return-deletion condition. Model effect may combine representation/training/
geometry; only a measured wrong-correspondence loss supports correspondence use.

Keep candidate only if clean MZ6 thin and far TP increase, each query FP is no
higher than MZ5, and wrong correspondence reduces recovered correct detections.
Report near recall and all wrong-near activations separately. Failed gate stops
this fit without tuning or new confirmation capture. Successful gate warrants
one separately frozen new-placement/background confirmation capture, up to200
samples with paired controls and configuration-level summaries, before retention.
No temporal reopening unless complete evidence detects, deletion loses detections,
and real observable history can restore them; this experiment tests only the
first two links. No history module is in scope.
