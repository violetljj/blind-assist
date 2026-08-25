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

An additional constant-prior audit further limits that interpretation.
`PRESERVE` was a valid mode for `1482/1806 = 82.06%` of validation pairs, so an
image-independent always-PRESERVE predictor was stronger than OA-V2. The
selected seed exceeded this prior by only `15` pairs or `+0.83` points, while
seed `1701` was `8` pairs or `-0.44` points below it. Moreover, `392/1806 =
21.71%` of pairs allowed both PRESERVE and FLIP and therefore carried no
PRESERVE/FLIP discrimination. On the remaining `1414` pairs, the selected arm
was `1105/1414 = 78.15%` versus `1090/1414 = 77.09%` for always-PRESERVE, a
gain of only `15` pairs or `+1.06` points. Thus task training had a positive
gain over frozen OA-V2, but its excess over the slot prior was small and did
not reproduce across both seeds.

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
relative to frozen OA-V2 on an under-target but house-disjoint collection. It
does not establish a seed-robust visual gain beyond the slot prior, reliable
owner-orientation recovery, the preregistered advancement effect, final-test
generalization, natural-scene, live-device, Android/default-App, product,
universal, or safety performance.
