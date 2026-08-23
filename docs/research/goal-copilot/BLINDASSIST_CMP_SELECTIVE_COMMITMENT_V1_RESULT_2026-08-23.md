# BlindAssist CMP selective-commitment V1 result

Status: `SEALED / FRESH_CONFIRMATION / SELECTIVE_COMMITMENT_NOT_SUPPORTED / NO_RERUN / NO_RESERVE / NO_P1`

## Frozen design

- Source and truth: the same official CMP Facade RGB/XML/PNG release and evaluator-private native door pixels used by
  the prior 89-image diagnosis.
- Fresh denominator: all 122 eligible images not present in the consumed 89-image roster, ranked before provider
  output and split into `32 Development / 64 Confirmation / 26 Reserve`.
- Overlap: 0 between all splits and 0 with the consumed 89; all 122 RGB hashes are unique.
- Providers: unchanged Grounding DINO proposal provider and unchanged Terra Brain. Teachers: 0.
- V1: an offline deterministic gate only. A raw `SELECT` becomes `CONTESTED` unless it passes the selected confidence
  and provider-rank requirements. Raw outputs are preserved and V1 adds zero provider calls.
- Six policies and all Development selection and Confirmation success rules were frozen before Development output.
  Retry/rerun was 0; interruption would be consumed as `in_doubt`.

The fresh roster file SHA-256 is
`55f4d2da77e4812aa7a7a52b8449804fe9761268a60a09d96de1f90331a389a4`.
The provider lock file SHA-256 is
`60350d4e312ba28a091bf554d2225f3e0ff518bbca0fcd7a53bc577bd48c72e5`.

## Development selection

Development completed 32 observations in four provider batches with `0 in_doubt` and no rerun. The predeclared
safety gates selected:

```text
CONF_075_RANK1
brain confidence >= 0.75
selected provider rank == 1
```

This policy made 12 commitments: 11 correct and 1 wrong, for `91.7%` commitment accuracy. The selection receipt was
sealed before any Confirmation provider call; its file SHA-256 is
`5df73994a5c3a56c455c4199ab59d0fda218ab227cd3a19b4551ef134563ef5f`.

## Fresh Confirmation result

Confirmation completed 64 observations in eight provider batches with `0 in_doubt`, no teacher, no retry, and no
rerun. Proposal availability at IoU 0.50 was `56/64`; Recall@1/3/5/10 was `25/49/54/56`.

| Metric | V0 | V1 |
|---|---:|---:|
| Raw/retained commitments | 43 | 16 |
| Correct grounding | 31 | 14 |
| Wrong confident guidance over all observations | 12/64 | 2/64 |
| Commitment accuracy | 31/43 = 72.1% | 14/16 = 87.5% |
| Correct-grounding retention | — | 14/31 = 45.2% |

V1 emitted 27 new `CONTESTED` states offline. Its final outcomes were
`14 CORRECT_GROUNDING / 8 PROPOSAL_MISS / 41 REFERENT_SELECTION_ABSTAINED_WITH_USABLE_PROPOSAL /
1 WRONG_CONFIDENT_GUIDANCE`. Although wrong commitment decreased and commitment accuracy increased, the frozen
success rule also required at least 80% correct-grounding retention. The observed `45.2%` fails that gate.

Formal verdict:

```text
SELECTIVE_COMMITMENT_NOT_SUPPORTED
```

The authoritative Confirmation result file SHA-256 is
`6d63f800be3fa1dcbc16025c1140ee2a2e8a285c6e1f9c4e93df97c1bb77e6fa`; its content SHA-256 is
`64f0f84569e581585aaeab03ad8ce69509eb36d68073b33a7d5a6b18928cf758`.

## Interpretation and closure

The experiment rejects this simple confidence-plus-rank-1 gate, not the broader need to model contested referents.
Provider rank is not referent authority: only `25/64` observations had a correct rank-1 proposal while `56/64` had a
usable proposal within the frozen candidate set. Treating rank 1 as the safe commitment boundary therefore removes
too many correct selections. The result does not authorize a threshold change, another policy from the observed grid,
Reserve activation, provider/prompt/goal replacement, or a rerun.

The consumed evidence still locates a current-frame selection/commitment problem in this constrained domain, but no
working V1 is established. A future successor requires a separately preregistered representation or verifier
hypothesis and new truth-bearing data; it cannot rescue this cohort. `LOST_AFTER_VISIBLE` remains unevaluated and P1
persistence remains unauthorized.

## Claim ceiling

`CMP_STATIC_GENERIC_DOOR_SELECTIVE_COMMITMENT_NEGATIVE_NO_NAMED_REFERENT_APPROACH_CONTROL_RANGE_BEARING_ARRIVAL_LOST_PERSISTENCE_SAFETY_OR_PRODUCT_CLAIM`

