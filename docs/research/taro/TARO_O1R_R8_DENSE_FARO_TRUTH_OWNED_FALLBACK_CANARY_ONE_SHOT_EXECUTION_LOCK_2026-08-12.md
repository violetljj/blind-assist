# TARO O1R R8 dense FARO truth-owned fallback canary lock

The consumed sparse ray-space V1 reached 54 clear labels across six parents but
failed compatibility: it reclassified 36 old occupied labels as clear and put 33
frozen positive-occupancy predictions on those clear labels. V1 remains failed.

This one-shot changes only the missing-query truth interface. Every source query
that already exists is evaluated by the exact prior dense FARO label algorithm.
When a source query is absent, the FARO support plane constructs the corresponding
fixed 3x3 query and the same dense obstacle, support, forward-observation, and
knownness logic evaluates it. `UNKNOWN` remains excluded from negatives.

The run reuses exactly the same eight parents and 133 FARO frames. It cannot read
unselected FARO, fit a selector or threshold, train, or promote R8. A pass only
authorizes freezing this interface before a separate fresh-parent confirmation.
