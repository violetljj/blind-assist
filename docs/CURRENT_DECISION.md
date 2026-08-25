# Current decision: GRAIL R1CL

Status: `ACTIVE / DEVELOPMENT`

## Question

Can a pairwise owner-coordinate objective on the frozen GRAIL R1CL data produce
a visible and measurable improvement over the credible existing baseline while
keeping the evaluator boundary intact?

## Fixed surface

- Code and contract: `research/active/grail-r1cl/`
- Local dataset: configured by `r1cl_train_dataset`
- Local DINOv2 backbone: configured by `r1cl_backbone`
- Runtime: configured by `research_python`
- Outputs: ignored `artifacts.local/`

Machine paths belong only in ignored `config/local.toml`. The tracked template
contains portable examples.

## Next action

Run the smallest controlled training/evaluation slice that can distinguish the
pairwise objective from the baseline. Record the metric and exact evidence
scope in this file only if it changes the decision.

## Stop condition

Stop or change the information source when the route shows no interpretable
gain under the frozen comparison. Do not rescue an information-poor route with
threshold, tracker, aggregation, or backbone sweeps.

## Claim ceiling

Environment smoke is complete. No training improvement, independent holdout,
natural-distribution, live-device, universal, or safety claim is currently
authorized by this route.
