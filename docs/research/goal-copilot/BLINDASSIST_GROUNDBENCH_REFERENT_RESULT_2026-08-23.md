# BlindAssist GroundBench referent result

Status: `SEALED / PUBLIC_DATASET_DERIVED_GT_STRONG / DOMAIN_LEXICON_PROPOSAL_UNION_NOT_SUPPORTED / NO_RERUN / NO_P1`

## Source and authority

- Metadata: GroundBench release `1.0.0` at `Social-AI-2026/GroundBench@010520d396f7b1775adc425e0b88fdc6fe95bb34`.
- Pixels: referenced COCO 2014 images, downloaded individually and accepted only after exact SHA-256 agreement with
  the released image manifest.
- Truth: RefCOCO-family expressions and GroundBench polygon coordinates derived from the source annotations. Target
  polygons and correct-candidate membership stayed evaluator-private.
- Eligible universe: 353 vehicle/outdoor-accessory records with at least one same-class distractor. All rosters were
  ranked and frozen mechanically; no person selected images or labels.
- Use ceiling: GroundBench original material is limited by its release notice to non-commercial research,
  evaluation, and peer review. COCO/RefCOCO-family content retains its own third-party terms; this work does not
  infer redistribution or commercial rights.

The pinned GroundBench 64-point benchmark SHA-256 is
`c111145c1ffc21a8821245755d0c7d8ef3218258d7e0d7ae2f36da8a9459ecf8`; the released image-manifest SHA-256 is
`224e8c984f8002cae96bc3bf2b9ce886ab59f4e53b18f0b8e7dd400be7ae7472`.

## Strong-truth baseline

The first 89 frozen observations used the unchanged Grounding DINO proposal provider and Terra Brain. All 89 pixels
passed released-hash verification. At IoU 0.50:

- proposal availability: `77/89`; Recall@1/3/5/10: `47/75/77/77`;
- actions: `79 SELECT / 5 AMBIGUOUS / 5 ABSTAIN`;
- outcomes: `65 CORRECT_GROUNDING / 12 PROPOSAL_MISS / 7 abstain-or-ambiguous-with-usable-proposal /
  5 wrong-with-usable-proposal`;
- all wrong commitments: `14/89`; commitment accuracy: `65/79 = 82.3%`.

The first 24 failures split evenly into `12 PROPOSAL_MISS` and `12 REFERENT_SELECTION / COMMITMENT`. This established
that both layers matter on static same-class referents; it did not establish an approach, control, or temporal result.

The roster SHA-256 is `891180a99f39d4b1f1c45bd1803be3977d34c9c30f7410c171f139920be04a86` and the report file SHA-256 is
`5affaaaa5087bec26f041357f333672afe2aca9a3b07d708d100893fb3e50455`.

## Fixed domain-lexicon successor

Failure inspection showed expressions naming visible parts such as a license plate, taillight, mirror, or ski part
while the private truth binds the parent COCO instance. The only authorized successor therefore added a frozen list
of 24 public COCO categories to the expression proposal pass and unioned the candidates. The Brain, weights,
thresholds, prompts, IoU rule, and success rule stayed fixed:

```text
proposal_v1 > proposal_v0
AND correct_v1 > correct_v0
AND wrong_all_v1 <= wrong_all_v0
```

The first 64-observation Confirmation attempt completed proposal generation and all V0 calls, but failed before the
first V1 process was created because the unchanged 46,284-byte prompt exceeded the Windows argument-length limit.
It was sealed `NOT_EVALUABLE_TRANSPORT_RUNTIME`; V1 provider calls were zero, and it carries no scientific verdict.
The only repair moved the identical prompt from argv to stdin on a fresh frozen cohort.

On the next 64 observations, the uncapped union increased proposal availability `54 -> 57` and correct grounding
`42 -> 44`, but also increased all wrong commitments `16 -> 19`; median candidate count rose `3 -> 14` and the maximum
reached 100. The frozen verdict was `DOMAIN_LEXICON_PROPOSAL_UNION_NOT_SUPPORTED`. That consumed result was used only
as Development evidence to choose `K=10`: the smallest recorded union cutoff whose Recall (`55`) exceeded V0's total
availability (`54`).

## Untouched bounded Top-10 Confirmation

The final 64 observations were frozen at positions 218--281 before model output. V0 used the expression pass; V1
used expression plus the same fixed 24-category union, truncated to Top-10 before the unchanged Brain. All 16 Brain
batches succeeded once via stdin; provider `in_doubt=0`, teachers=0, retries=0, and reruns=0. Public provider inputs
contained no private truth literals.

| Metric | V0 | V1 Top-10 |
|---|---:|---:|
| Proposal availability | 57/64 | 59/64 |
| Recall@1 / @3 / @5 / @10 | 37 / 56 / 57 / 57 | 28 / 50 / 56 / 59 |
| Correct grounding | 47 | 46 |
| Proposal miss | 7 | 5 |
| Abstain/ambiguous with usable proposal | 3 | 2 |
| Wrong with usable proposal | 7 | 11 |
| All wrong commitments | 13/64 | 16/64 |
| Selection accuracy given usable proposal | 47/57 = 82.5% | 46/59 = 78.0% |
| Commitment accuracy | 47/60 = 78.3% | 46/62 = 74.2% |
| Candidate count median / maximum | 3 / 8 | 10 / 10 |

The bounded union recovered two additional usable proposals but lost one correct end-to-end grounding and added three
wrong commitments. It therefore fails two clauses of the frozen success rule. Formal verdict:

```text
DOMAIN_LEXICON_PROPOSAL_UNION_NOT_SUPPORTED
```

The final roster SHA-256 is `55686533ac089599056cef866e8d216f9ece921a00d606eb0329b16a1e44934e`. The report file SHA-256 is
`66a6f632818948c3fbdade9306640c7415d20299ce082a8e7d27fe46bfe6584f`; its content SHA-256 is
`f164bf2701688c80d1df9ae56f60ff89f044016f3de7897dbedc51a0743c3a4a`.

## Interpretation and closure

A fixed class-lexicon union, even bounded to ten candidates, is not a supported V1 for this Brain. Proposal coverage
alone is insufficient: extra candidates degrade referent selection and increase confident wrong guidance. The result
does not show that all proposal work is futile, but it rejects category-union expansion without a candidate-level
referent verifier or representation that can exploit the added recall safely.

No same-cohort threshold, prompt, provider, goal, lexicon, K, or teacher change is permitted. The remaining 72 reserve
identities stay untouched. This static COCO result neither evaluates target acquisition in an approach episode nor
range, bearing, arrival, `LOST_AFTER_VISIBLE`, persistence, safety, blind-user effectiveness, or product readiness.
P1 remains unauthorized.

## Claim ceiling

`STATIC_COCO_OUTDOOR_REFERENT_SELECTION_ONLY_NO_APPROACH_CONTROL_RANGE_BEARING_ARRIVAL_LOST_PERSISTENCE_SAFETY_OR_PRODUCT_CLAIM`
