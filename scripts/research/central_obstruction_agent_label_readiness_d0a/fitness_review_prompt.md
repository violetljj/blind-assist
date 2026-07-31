# D0-A0 current-task fitness canary

Inspect exactly the frozen, evenly spaced RGB frames named in the input bundle. For each proposed
production-labeling session, decide only whether:

1. the central image area is visibly present often enough for a later D0-A1 image-space ROI;
2. frame content and quality are sufficient to support later Agent observation review with explicit
   `NOT_EVALUABLE` states;
3. continuous parent-event grouping appears possible from the full ordered session without any
   detector, candidate-model, truth, risk, feedback, or prior effect output.

Do not label obstruction events, choose an ROI, set matching rules or numeric readiness thresholds,
infer intended route/traversability/safety, or inspect candidate output. This is a bounded D0-A0
source-fitness canary, not D0-A2 production annotation.
