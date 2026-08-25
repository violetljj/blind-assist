# Current decision: GRAIL R1CL

Status: `STOP_R1C_L_WITHOUT_FINAL_TEST / DEVELOPMENT_GATE_NOT_MET / FINAL_UNOPENED`

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

## Development result

The single architecture and both frozen seeds completed on the house-disjoint
Development collection. Seed `2701`, epoch `3` was selected at
`1497/1806 = 82.89%` validation slot accuracy. The frozen paired-relative
OA-V2 baseline on the identical pairs was `1456/1806 = 80.62%`, so R1C-L
improved by only `41` pairs or `+2.27` percentage points. This is below the
frozen `+8` point gate. Doorway improved by `+4.29` points and Drawer by
`+1.82` points, so the effect was positive in both classes but not large enough
to advance.

Train and validation collection were transparently under their approximate
pair targets: `12726/20000` and `1806/2000`. Training was nevertheless run as
Development evidence; this shortfall limits precision but is not a reason to
erase the observed comparison. Exact hashes and denominators are in
`research/active/grail-r1cl/grail_r1c_l_development_result_v1.json`.

## Next action

Do not access the final-test data or rerun/tune R1C-L on this consumed cohort.
Any successor must be separately authorized and must change the information
source rather than sweep this RGB-only model.

## Stop condition

Met: the validation uplift was below `+8`, so this route stops without final
access. Do not rescue it with threshold, tracker, aggregation, backbone, crop,
bin, loss-weight, seed, or ensemble sweeps.

## Claim ceiling

This establishes only a small positive synthetic ProcTHOR Development effect
on an under-target but house-disjoint collection. It does not establish the
preregistered advancement effect, final-test generalization, natural-scene,
live-device, Android/default-App, product, universal, or safety performance.
