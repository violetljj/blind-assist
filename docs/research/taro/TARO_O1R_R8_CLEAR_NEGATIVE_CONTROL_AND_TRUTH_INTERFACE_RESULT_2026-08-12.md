# TARO O1R R8 clear negative-control and truth-interface result

R8 is closed as valid but `NOT_EVALUABLE`, not as an algorithm failure or a
success. On the selected 8 parents / 133 frames / 1,197 queries, the frozen dense
labels contained only 3 definite-clear queries across 3 parents, below the fixed
50-query / 4-parent negative-control floor.

The positive-occupancy evidence remains strong but descriptive: precision was
1.0, recall 0.9683, one-sided Wilson precision lower bound 0.9967, and parent-
macro definite-occupancy coverage improved by 0.9792. None of this promotes the
route because the clear denominator is inadequate.

Two bounded truth-interface attempts were preserved:

- sparse ray-space V1 reached 54 clear queries across 6 parents, but missed dense
  obstacles: 36 old occupied labels became clear and 33 frozen positive
  predictions landed on those labels. V1 is invalid as a replacement labeler;
- dense truth-owned fallback V2 retained all 854 old definite labels exactly,
  but all 198 missing-query slots remained `UNKNOWN`. Those slots came from
  frames whose FARO support plane was also unevaluable, so constructing a body/
  path query would require inventing ground geometry.

Among the remaining 145 old `UNKNOWN` queries with a source query, only 39 were
otherwise obstacle-free with support and full 2 m forward coverage. Even an
optimistic visibility-only admission would yield 42 clear queries total, still
below 50. Further interface tuning on consumed R8 is therefore stopped.

The unique successor is development-only labeling of the remaining 16 already
downloaded and user-authorized R8 parents. That evidence may be used to freeze a
truth-blind source-only clear-enrichment rule, but every R8 parent is thereafter
consumed. Any confirmation must use entirely fresh parents. `UNKNOWN` remains
excluded from negatives, and there is no deployment, product, or safety claim.
