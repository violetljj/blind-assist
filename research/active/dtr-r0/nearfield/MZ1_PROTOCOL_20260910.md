# MZ1: tiny learned fusion with frozen visual features

2026-09-10 EXPLORE, conditional on the successful MZ0 information comparison.
Use the same admitted5000 source and original roles:2500 TRAIN_ONLY,1000 DEV_ONLY,
1500 EVAL_ONLY. All are consumed controlled Development. No new RGB backbone,
augmentation, source selection or threshold tuning. Preserve original B alerts.

Cache the frozen JOINT decoder's normalized12x64 visual input plus its four range
probabilities (772 features). Cache clean8x8 two-surface ranges/4 and validity
(256 features) with the unchanged MZ0 observation operator. No target masks,
true physical class, scene identity or evaluator counts are input features.
Native counts supply training targets/evaluation only. Missing range uses0 plus
an explicit validity mask; zero prediction never asserts safety clearance.

Three models have identical1028->128->4 MLP structure, ReLU, initial weights
(seed53), training indices (seed59), and training settings. RGB_ONLY zeros the
ToF slice, TOF_ONLY zeros the visual slice, FUSION uses both. AdamW learning
rate0.001, weight decay0.0001, BCEWithLogits, batch128, exactly300 steps. Only
TRAIN_ONLY targets are used for fitting. No best-step selection or random-seed
search. Final DEV/EVAL scores at fixed0.5 threshold; no holdout-driven changes.

Compare final EVAL exact four-event accuracy, wrong-far and cross-body errors
with both matched single-input controls and MZ0's fixed two-surface readout.
The learned-fusion contribution requires exact at least2pp above both MZ0 and
RGB_ONLY, no worse than TOF_ONLY, no more wrong-far or combined cross-body errors
than MZ0, and all original alerts unchanged. If ToF alone explains the gain,
retain the information-source result without claiming fusion-specific novelty.

Stop after one three-arm fit, final scoring, independent checks and delivery.
No continuation fitting or stronger architecture to rescue a failed criterion.
MZ2 remains a distinct resolution/corruption comparison after interpreting these
results. No real sensor, natural-scene or safety-performance claim.
