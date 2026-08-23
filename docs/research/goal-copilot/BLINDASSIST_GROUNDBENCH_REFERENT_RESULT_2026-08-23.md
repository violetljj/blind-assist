# BlindAssist GroundBench referent result

Status: `SEALED / PUBLIC_DATASET_DERIVED_GT_STRONG / PROPOSAL_UNION_NOT_SUPPORTED / RELATIONAL_RANKER_NOT_SUPPORTED / NO_RERUN / NO_P1`

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

## Candidate-level CLIP verifier Development

After the proposal-union closure, a distinct candidate-level representation was tested without consuming fresh
Confirmation data. The already-consumed 89-observation baseline was designated Development; its expression-only
proposal boxes stayed unchanged. A pinned local `openai/clip-vit-base-patch32` scored four representations frozen
before scoring: exact crop, 1.25-expanded crop, focused whole-image context, and the mean of expanded crop plus focused
context. Selection was maximum Recall@1, then MRR, then fixed variant order. No Brain or teacher was called.

| Development ranking | Recall@1 | Recall@3 | Recall@5 | MRR given usable |
|---|---:|---:|---:|---:|
| Original provider order | 47 | 75 | 77 | 0.7807 |
| Exact crop | 47 | 71 | 76 | 0.7688 |
| Expanded crop | 41 | 72 | 77 | 0.7307 |
| Focused context | 44 | 65 | 73 | 0.7223 |
| Expanded crop + focused context | 44 | 69 | 77 | 0.7470 |

The selected exact-crop variant tied rather than exceeded provider Rank@1 and reduced MRR. It therefore failed the
predeclared gate and was sealed `CLIP_CANDIDATE_VERIFIER_DEVELOPMENT_NOT_PROMISING`. Under that zero-shot hypothesis,
positions 282--345 were not frozen or downloaded, Confirmation provider calls remained zero, and all 72 remaining
identities stayed untouched. This rejects zero-shot CLIP crop/focused-context reranking, not learned relational
verification in general.

## Learned relational-ranker Confirmation

A separate fixed ranker then combined provider score/rank, the frozen CLIP scores, normalized candidate geometry,
and fixed expression-relation interactions such as left/right/top/bottom. Pairwise logistic regression with 21 fixed
features and no hyperparameter search trained only on consumed positions 1--217. Consumed positions 218--281 were a
held-out Development block: on 57 usable-proposal observations, Rank@1 improved `37 -> 41` and MRR improved
`0.8026 -> 0.8509`, passing the frozen authorization gate.

Only then were fresh positions 282--345 frozen and their 64 COCO pixels hash-verified. V0 and V1 used exactly the same
expression-only candidates; V1 changed only candidate order using the frozen ranker. All 16 Brain batches succeeded
once through stdin, with `in_doubt/teacher/retry/rerun = 0/0/0/0`; public inputs had zero private-truth literal hits.

| Fresh Confirmation metric | V0 | Relational ranker V1 |
|---|---:|---:|
| Proposal availability | 52/64 | 52/64 |
| Recall@1 / @3 / @5 / @10 | 33 / 48 / 51 / 52 | 33 / 48 / 50 / 52 |
| MRR given usable proposal | 0.7736 | 0.7877 |
| Correct grounding | 41 | 42 |
| Wrong confident guidance over all observations | 20/64 | 18/64 |
| Selection accuracy given usable proposal | 78.8% | 80.8% |
| Commitment accuracy | 67.2% | 70.0% |

The end-to-end direction was mildly favorable, but the predeclared candidate mechanism did not reproduce: Rank@1
was tied `33 -> 33`. The strict success rule required Rank@1 improvement in addition to more correct grounding and no
increase in wrong guidance. Formal verdict:

```text
RELATIONAL_CANDIDATE_RANKER_NOT_SUPPORTED
```

The roster SHA-256 is `a0c16097f7d234089c9aa489bcc9a9cd702a2c7507c3b6184912f2756c51812a`. The report file SHA-256 is
`3e3ad6b9e1fc190527733bc42f03112b2ff9b47ebbfe0ddf50d61d46e0966c3f`; its content SHA-256 is
`671b3abd1239b763fd6d4551a21c059657f7803530cdd699beb3236290937445`. The final eight source identities remain
untouched; they cannot be used to relax the gate or rescue this cohort.

## Claim ceiling

`STATIC_COCO_OUTDOOR_REFERENT_SELECTION_ONLY_NO_APPROACH_CONTROL_RANGE_BEARING_ARRIVAL_LOST_PERSISTENCE_SAFETY_OR_PRODUCT_CLAIM`
