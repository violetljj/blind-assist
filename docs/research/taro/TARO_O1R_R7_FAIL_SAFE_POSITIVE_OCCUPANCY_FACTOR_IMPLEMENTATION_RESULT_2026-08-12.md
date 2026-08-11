# TARO O1R R7 fail-safe positive-occupancy factor implementation

Status: `IMPLEMENTED_RESEARCH_ONLY_NOT_PROMOTED`.

The frozen R7 positive rule is now an independent sealed factor API. Its public
builder has no FARO, truth, label, or outcome input. It can emit only
`OCCUPIED_OBSERVED` or `UNKNOWN`; historical clear states and absence of
positive evidence are deliberately mapped to `UNKNOWN`.

Six focused tests pass, including tamper-to-clear rejection. A source-only
replay across all 170 fresh Phase-A frames reproduced 1,292 occupied and 238
unknown query states with zero clear outputs.

This is executable research code, not a promoted effectiveness result. The R7
fresh terminal remains dual-class `NOT_EVALUABLE`; the next task is a separately
frozen source-only clear-negative-control cohort-enrichment protocol.
