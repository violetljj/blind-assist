# MZ54: matched full-RGB raster result

Primary DROP_CLOSE gate: **FAIL**. Nonfit BODY_NEAR TP is FULL 41 versus matched CROP 41; nonfit total FP is 20/21, old noncalibration FP is 31/39. The gate tests this query and FP totals; it does not imply improvements in every query or profile.

| DROP_CLOSE group | matched CROP TP/FP/FN | FULL TP/FP/FN | FULL_CROP_ONLY TP/FP/FN | fixed MZ53 OLD_NEG union TP/FP/FN |
| --- | ---: | ---: | ---: | ---: |
| legacy_noncal | 2712/39/236 | 2710/31/238 | 2710/31/238 | 2722/45/226 |
| mz48_fit | 644/38/556 | 592/39/608 | 581/39/619 | 743/38/457 |
| mz48_nonfit | 401/21/591 | 384/20/608 | 382/20/610 | 463/20/529 |
| all_noncal | 3757/98/1383 | 3686/90/1454 | 3673/90/1467 | 3928/103/1212 |

Old noncalibration excludes DEV. New nonfit is 640 heldout-site + 384 withheld-family. all_noncal includes fit 1280 once and is not wholly held-out evidence.

| nonfit DROP_CLOSE query | CROP TP/FP/FN | FULL TP/FP/FN | FULL_CROP_ONLY TP/FP/FN | fixed OLD_NEG union TP/FP/FN |
| --- | ---: | ---: | ---: | ---: |
| BODY_NEAR | 41/1/183 | 41/0/183 | 39/0/185 | 66/0/158 |
| BODY_FAR | 146/5/110 | 146/5/110 | 146/5/110 | 178/5/78 |
| HEAD_NEAR | 161/5/95 | 145/5/111 | 145/5/111 | 165/5/91 |
| HEAD_FAR | 53/10/203 | 52/10/204 | 52/10/204 | 54/10/202 |

| profile nonfit | CROP total TP/FP | FULL total TP/FP | FULL_CROP_ONLY total TP/FP | fixed OLD_NEG union total TP/FP |
| --- | ---: | ---: | ---: | ---: |
| IDEAL | 584/33 | 567/31 | 566/31 | 655/36 |
| MERGE_CLOSE | 626/21 | 622/18 | 620/18 | 668/21 |
| DROP_CLOSE | 401/21 | 384/20 | 382/20 | 463/20 |

| DROP_CLOSE FULL versus | group | TP gained/lost | FP added/removed | exact frames gained/lost |
| --- | --- | ---: | ---: | ---: |
| MZ54/CROP_RASTER/candidate | legacy_noncal | 0/2 | 2/10 | 10/4 |
| MZ54/FULL_CROP_ONLY/candidate | legacy_noncal | 0/0 | 0/0 | 0/0 |
| OLD_NEG/UNION | legacy_noncal | 0/12 | 2/16 | 14/11 |
| MZ54/CROP_RASTER/candidate | mz48_nonfit | 2/19 | 0/1 | 3/11 |
| MZ54/FULL_CROP_ONLY/candidate | mz48_nonfit | 2/0 | 0/0 | 2/0 |
| OLD_NEG/UNION | mz48_nonfit | 2/81 | 0/0 | 2/60 |
| MZ54/CROP_RASTER/candidate | mz48_fit | 11/63 | 1/0 | 11/48 |
| MZ54/FULL_CROP_ONLY/candidate | mz48_fit | 11/0 | 0/0 | 11/0 |
| OLD_NEG/UNION | mz48_fit | 10/161 | 1/0 | 10/112 |

| FULL recovered versus matched CROP | TP | native winner | native outside 45 degrees | native below original crop | bottom-boundary block |
| --- | ---: | ---: | ---: | ---: | ---: |
| BODY_NEAR | 2 | 2 | 2 | 2 | 0 |
| BODY_FAR | 0 | 0 | 0 | 0 | 0 |
| HEAD_NEAR | 0 | 0 | 0 | 0 | 0 |
| HEAD_FAR | 0 | 0 | 0 | 0 | 0 |

Below the original crop means winning center y>=292 (raster row>=37). Row 36 overlaps the crop boundary and is reported separately. Saved native-winning flags were independently looked up in the existing MZ52 uint8 count arrays; no native depth was opened. Correct events without a native winning flag are not automatically local evidence.

Validation: 562 prior MZ51 arrays exactly equal; 1680 individual/union count rows match sealed MZ51/MZ53 results; 251,424 independent scalar known-bit checks; 92,160 native-winning lookups. The exact prior schedule, train membership, 1250 shared unique fit frames, 1280 pairs and 2560-frame partition are preserved. All MZ37 positives and MZ36's 400 attempts/80 UNKNOWN remain. Only the two existing cutoffs were recomputed for parity; FULL_CROP_ONLY reuses FULL cutoff.

Attribution limit: FULL changes local RGB field and local-supervision field together. The matched control has 756 contained raster blocks, not the legacy 3136 angular samples. FULL_CROP_ONLY keeps full image features, context and fitted weights; it diagnoses pooling support and is not an identical-input crop ablation. Query/profile losses remain costs even if the narrow BODY_NEAR gate passes. No natural-scene, physical VL53L8CX calibration, clearance or default-App claim follows.

Evidence: `F:\ba-data\blindassist-artifacts-20260805\work\mz54-full-rgb-20260911/score-v1/result.json`, `audit.json` and final `receipt.json`. The final receipt binds this report, implementation, brief, source receipts and saved outputs.
