# TARO O1R R7 fit-only LOPO canary result

Machine-readable result: [JSON](TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_FIT_LOPO_CANARY_RESULT_2026-08-12.json).

The exact one-shot executed successfully on 8 `ADAPTER_FIT` parents, 211 frames and 1,899 queries. All eight leave-one-parent-out folds selected the same source-only positive-occupancy tuple. The R7 reducer produced 1,619 `OCCUPIED_OBSERVED` and 280 `UNKNOWN`, versus 1,899 `UNKNOWN` from the frozen R6 baseline. Against definite FARO labels it recovered 1,438/1,450 occupied queries with zero false occupied against a definite clear label.

The frozen fit gates therefore pass, but this is not a dual-class or deployment result. FARO produced zero identifiable `CLEAR_OBSERVED` labels and R7 produced zero clear predictions. The reported precision excludes 181 predictions whose FARO truth is `UNKNOWN`; it has no definite-clear negative denominator. Only the positive-occupancy hypothesis advances to a fresh-parent confirmation protocol. The far-censored clear branch remains not evaluable and unauthorized.
