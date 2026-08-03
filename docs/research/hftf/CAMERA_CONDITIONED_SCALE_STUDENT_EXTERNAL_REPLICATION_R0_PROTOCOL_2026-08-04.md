# Camera-Conditioned Scale Student External Replication R0

Date: 2026-08-04

Status: `FROZEN_BEFORE_EXTERNAL_STUDENT_PREDICTION_OR_EFFECT_EXECUTION`

The five-parent leave-one-parent-out result passed every frozen gate, so this protocol performs one fixed external replication before any feature, model, alpha, scale range, or effect gate can change.

The final ridge student trains once on all valid labels from the original five-parent consumed Development corpus. It is then evaluated on ten height-eligible parents (330 frozen anchors) from two existing TartanGround corpora. These corpora were consumed by earlier research, so they are not called globally fresh. They were not used to design, select, cross-validate, or train this student and are therefore `STUDENT_UNSEEN` external replication data.

Parent eligibility uses only the pre-existing `robot_height` metadata and the already frozen `[0.8, 2.2] m` receipt range. The exact parents, heights, corpus sample hashes, model/protocol hashes, and record count are frozen in the adjacent JSON. Student predictions cannot read external sensor depth or clearance truth. Those sources are joined only by the evaluator after prediction generation.

The R0 effect gates remain unchanged. No result-dependent rescue is allowed. Passing remains synthetic retrospective replication evidence, not a real-phone, wearable, independent-assistance, production, or safety claim.
