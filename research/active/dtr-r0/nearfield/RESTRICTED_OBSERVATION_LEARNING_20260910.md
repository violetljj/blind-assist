# Learning applied to restricted ToF, richer objects and compact data

2026-09-10. Target: useful BODY/HEAD near/far decisions from imperfect RGB and
VL53L8CX-oriented observations, with efficient collection and storage. Good
performance under restricted sensing is a meaningful research objective;
method novelty requires comparison with existing work, not just adding noise.

| Primary source inspected | Relevant finding | Decision for this project |
|---|---|---|
| [DELTAR, ECCV 2022](https://zju3dv.github.io/deltar/) | Combines RGB with zone-level low-resolution ToF distributions; the sensor region is not a known pixel-depth location. | Retain within-zone ambiguity. Do not invent variance or signal strength from privileged native depth and call it a device output. Our task is obstacle decisions, so dense-depth reconstruction is an optional comparator, not a required intermediate. |
| [MMRNet](https://arxiv.org/abs/2210.10842) | Its dynamic ensemble training includes both modalities, RGB only and depth only. Its experiments separate novel objects from the other splits. | Matched training should expose declared restricted observations; a gate alone may not learn failure behavior. Preserve asset identity to enable later held-out-object comparisons. Do not copy its invalid-depth inpainting as observed ToF evidence. |
| [UMoE](https://arxiv.org/html/2307.16121v1) | Motivates modality-specific uncertainty encoding because uncertainty scales and responses differ across sensors. | A shared selector score is not automatically calibrated branch correctness. Test learned quality representations against the simple existing fusion under the same inputs; borrowed uncertainty estimators also need cost measurement. |
| [WILDS, ICML 2021](https://proceedings.mlr.press/v139/koh21a.html) | Separates distribution shifts across domains from ordinary in-distribution evaluation. | Split richer data by mesh asset/family and source site where testing transfer; random frame splits alone cannot demonstrate generalization to new forms or backgrounds. |

These are mechanisms and evaluation ideas, not imported performance guarantees.
The sources use different sensors, tasks and datasets. This targeted reading is
not an exhaustive novelty search. Full-text MMRNet sections3.2/4.1 and the UMoE
motivation were inspected; DELTAR author-page and WILDS publisher material were
also inspected. Repeated search URLs were not counted as additional papers.

MZ40 shows frozen ideal-trained fusion degrades under constrained packets.
[MZ41](MZ41_MISSING_GUARD_20260910.md) then shows binary query-level missingness
alone trades four recovered true bits for four false bits. Thus the next model
question is matched representation/quality learning, rather than another
hand-picked zero-return threshold. This remains a hypothesis until tested.

Data work proceeds alongside that question: actual mesh rendering, unchanged
materials and native-surface labels distinguish shape variation from cosmetic
replacement. Save mesh/family/site/condition identities and observed relation;
retain mismatched attempts. A small one-site block establishes the pipeline,
not cross-scene robustness. Compact source storage preserves original bytes,
including invalid depth and support values, and permits direct package reads.
Quantizing depth or changing precision would be a separate experiment.
