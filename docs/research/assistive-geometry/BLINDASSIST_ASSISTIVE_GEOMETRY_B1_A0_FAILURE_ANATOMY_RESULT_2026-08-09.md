# B1-A0 failure anatomy result

Terminal: `B1_A0_FAILURE_ANATOMY_COMPLETE_NOT_ELIGIBLE_FOR_PROMOTION`

Eligibility: `NOT_ELIGIBLE_FOR_PROMOTION`

The three SHA-bound seeds fail in the same way. Of 1,139 truth-clear cells per seed, 841 / 852 / 870 are predicted occupied (73.84% / 74.80% / 76.38%). Every false block is internally consistent with a predicted-clearance threshold crossing; there are zero state/clearance aggregation inconsistencies and zero predicted-ground-invalid state inconsistencies. A0 reads no assistive occupancy/task head, so head collapse is not an admissible explanation.

Paired clearance is systematically conservative: signed bias is -0.216 / -0.226 / -0.256 m, with 65.32% / 64.89% / 69.86% negative residuals. False-block mask Jaccard is 0.924–0.936 and keyed residual correlation is 0.951–0.970 across seed pairs. Even at more than 0.50 m truth-clear margin above the affected horizon, 95 / 88 / 104 cells false-block. This localizes failure before the deterministic threshold, but aggregate observations cannot causally distinguish dense-depth scale, numeric ground/support error or another upstream geometry error.

Transition disagreement is predominantly persistent, not jitter: 797 / 804 / 818 events are stable truth-clear followed by stable predicted-occupied, representing 81.87%–82.33% of all transition failures. Only 45 / 47 / 61 are prediction flips on stable truth. Depending on whether proximity is measured against truth only or either truth/prediction clearance, 21.94%–22.51% or 37.07%–46.59% of failures are within 0.10 m of a horizon; 517 / 608 / 628 remain far from both.

Critical support boundary: all 1,139 defined truth-clear cells occur in parent `464241`; the other three parents have no truth-clear denominator. The result is strong evidence of a cross-seed-repeatable failure on that observed support, not an all-scene prevalence estimate.

B1-A0 remains permanently frozen. A1–A4, A0 repair/relabeling, and Geometry R2 promotion are not authorized. A materially different R2 would require a new pre-outcome protocol and disjoint selection evidence. Development Calibration and Confirmation remain sealed.
