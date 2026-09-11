# MZ75: the far-return comparison was confounded by slot representation

Moving the **same surviving far distances** from slot 1 to slot 0 materially changed the frozen pipeline. DIVERSE improved from 760 TP / 83 FP to 912 / 56 on MZ67, and from 881 / 106 to 987 / 31 on MZ61. No distance, valid-zone set, RGB, weight or cutoff changed. MZ74's FAR-versus-CLOSEST difference therefore cannot be attributed purely to which physical return survives. This is a consumed Development representation diagnostic, not a promotion or hardware result.

The test used exactly 1,024 original HELDOUT_GEOMETRY frames per source, following [the fixed protocol](MZ75_RETURN_SLOT_CONTROL_20260911.md). Each source has 4,096 query bits: 1,024 positive and 3,072 known-negative bits; each individual query has 256 positives and 768 negatives. Query UNKNOWN is zero in this subset; this does not mean all native cells are known. Packing changed 11,740 zones in 988 MZ61 frames and 9,250 zones in 903 MZ67 frames. All other zones remained byte-identical, and every zone's valid-distance multiset and availability remained identical.

All values below are the final OLD_NEG OR MZ70 candidate decision, with original fixed cutoffs. Columns show TP / FP / FN.

| Source | Arm | Query | FAR in slot 1 | Same FAR packed into slot 0 |
|---|---|---|---:|---:|
| MZ61 | CONTROL | BODY_NEAR | 251 / 0 / 5 | 238 / 0 / 18 |
| MZ61 | CONTROL | BODY_FAR | 236 / 11 / 20 | 254 / 3 / 2 |
| MZ61 | CONTROL | HEAD_NEAR | 256 / 39 / 0 | 256 / 4 / 0 |
| MZ61 | CONTROL | HEAD_FAR | 157 / 73 / 99 | 239 / 26 / 17 |
| MZ61 | DIVERSE | BODY_NEAR | 249 / 0 / 7 | 234 / 0 / 22 |
| MZ61 | DIVERSE | BODY_FAR | 236 / 11 / 20 | 255 / 3 / 1 |
| MZ61 | DIVERSE | HEAD_NEAR | 253 / 22 / 3 | 256 / 4 / 0 |
| MZ61 | DIVERSE | HEAD_FAR | 143 / 73 / 113 | 242 / 24 / 14 |
| MZ67 | CONTROL | BODY_NEAR | 208 / 0 / 48 | 208 / 0 / 48 |
| MZ67 | CONTROL | BODY_FAR | 187 / 23 / 69 | 232 / 26 / 24 |
| MZ67 | CONTROL | HEAD_NEAR | 246 / 46 / 10 | 233 / 11 / 23 |
| MZ67 | CONTROL | HEAD_FAR | 82 / 28 / 174 | 135 / 19 / 121 |
| MZ67 | DIVERSE | BODY_NEAR | 232 / 0 / 24 | 226 / 0 / 30 |
| MZ67 | DIVERSE | BODY_FAR | 187 / 23 / 69 | 232 / 26 / 24 |
| MZ67 | DIVERSE | HEAD_NEAR | 246 / 32 / 10 | 233 / 11 / 23 |
| MZ67 | DIVERSE | HEAD_FAR | 95 / 28 / 161 | 221 / 19 / 35 |

The improvement includes losses and new false bits, not just favorable net totals:

| Source / arm | Total TP / FP, slot 1 → packed | TP gained / lost | FP added / removed |
|---|---|---:|---:|
| MZ61 CONTROL | 900 / 123 → 987 / 33 | 101 / 14 | 4 / 94 |
| MZ61 DIVERSE | 881 / 106 → 987 / 31 | 121 / 15 | 3 / 78 |
| MZ67 CONTROL | 723 / 97 → 808 / 56 | 118 / 33 | 9 / 50 |
| MZ67 DIVERSE | 760 / 83 → 912 / 56 | 181 / 29 | 9 / 36 |

In particular, packing loses near-body TP on MZ61 and near-head TP on MZ67 while recovering far-query TP. Of DIVERSE's newly gained TP, 33 on MZ61 and 75 on MZ67 are already positive in the changed OLD_NEG path; 88 and 106 respectively require the MZ70 head beyond that path. These are decision-path partitions, not causal native-winner or feature attribution. Native argmax and winner correctness were not re-evaluated in this control.

With both single surviving returns now represented in slot 0, CLOSEST versus packed FAR is much closer descriptively:

| Source / arm | CLOSEST TP / FP | Packed FAR TP / FP |
|---|---:|---:|
| MZ61 CONTROL | 987 / 33 | 987 / 33 |
| MZ61 DIVERSE | 988 / 31 | 987 / 31 |
| MZ67 CONTROL | 805 / 57 | 808 / 56 |
| MZ67 DIVERSE | 911 / 57 | 912 / 56 |

These totals still hide query-level exchanges and do not establish interchangeable return physics. The previous TRAIN-only audit found zero slot-1-only zones across each MZ70 arm's 4,194,304 scheduled zone presentations, whereas several consumers have explicit ordered range/valid channels. MZ75 now demonstrates a substantial representation effect at fixed observed distances; it does not establish real thin-target survival, strongest-target behavior or a deployable sensor model. Both arms and all eight decision outputs remain reported; no winner is promoted.

Execution and checks: each source ran 16 original TRAIN parity frames immediately before its 1,024 HELD frames. All 32 parity frames passed raw tolerance 2e-5 / 1e-6 and exact decision signs (52 comparisons); 2,048 HELD frames completed. The implementation interleaves parity and inference by source, rather than completing both sources' parity before any new inference. Both parity checks were required for a PASS receipt, and no outcomes changed the fixed run. There were 2,080 RGB loads; the three existing visual views were shared across the heads. The run receipt records 42.4327173 s, versus 50.2852483 s for the launch wrapper. Independent CPU scoring took 4.6208079 s in its receipt (4.621 s rounded). It checked 262,144 packet scalars, 32,768 candidate scalars and 327,680 metric bits. Both predictor and scorer exited with code 0 and no remaining owned processes; compact handles were released and no permanent dense cache was created. No fitting, new cutoffs or native-depth decoding occurred.

[Full eight-method metrics and original IDEAL/CLOSEST/DROP references](../../../../artifacts.local/work/mz75-return-slot-control-20260911/score-v1/result.json), [paired event IDs](../../../../artifacts.local/work/mz75-return-slot-control-20260911/score-v1/paired-events.json), [scalar audit](../../../../artifacts.local/work/mz75-return-slot-control-20260911/score-v1/audit.json), [run receipt](../../../../artifacts.local/work/mz75-return-slot-control-20260911/run-v1/receipt.json), [score receipt](../../../../artifacts.local/work/mz75-return-slot-control-20260911/score-v1/receipt.json), [execution](../../../../artifacts.local/work/mz75-return-slot-control-20260911/execution-v1.json), [score execution](../../../../artifacts.local/work/mz75-return-slot-control-20260911/score-execution-v1.json).

SHA-256 bindings:

- Run receipt: `72bddc2b0825edcbe733c05ea8b49f44325c2235151404400b6b096cbdbf7b1d`.
- Score receipt: `f44044e744b9c3f49b92ce33c6bca69bf2ad7c5b486fd4d2fc43210caa83fe90`.
- Result: `bf797b8001793fe98a9a01194c0813ccb0ca6cfa09a7ee4ad27504518e0ab2f6`.
- Paired events: `ec1d22337cf05c1cbf06c60f4e207809223d4fe44016275722a9c3a157869bf5`.

Separately, [actual-surface anchor work](SOURCE_SURFACE_ANCHORS_20260911.md) prepared source-geometry repair candidates. That CPU geometry work produced no new capture or source admission and is not evidence from this model comparison.
