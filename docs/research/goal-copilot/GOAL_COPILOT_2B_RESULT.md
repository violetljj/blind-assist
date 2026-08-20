# GOAL-COPILOT-2B result

Terminal status: `GOAL_COPILOT_2B_SEARCH_COMPLETE_NO_HELDOUT_ADMISSION`.

Verdict: `GC2B_NOISE_ROBUST_SEARCH_SIGNAL_NOT_ESTABLISHED`.

Both frozen Sky replicates completed their full budgets. No retry, replacement,
resume, GC1 fresh access, or hidden/held-out evaluation occurred during search.

## Frozen execution identity

- BlindAssist commit: `f3aefccbb874fe2d2479a7263dd5283569a17dc8`;
- SkyDiscover commit: `a2692f009cf97c4b2da4b70674f780fb39f5bf23`;
- SearchTaskBundle digest:
  `766e41c3f6de6e3c856265af609381c1a49920e6ad59d195605eca60b25f3666`;
- protocol seal digest:
  `b9de3917574d5e0a309205134640875561dfbc3295118124ca229ff0970e1286`;
- provider: `E:/codex-tools/bin/codex.exe`, `codex-cli 0.148.0`, executable
  SHA-256 `2ad2cf8a732da68b8f141634f92db1a03016c5faf533a7225fbc0fb740130410`.

## Budget and candidates

- replicate 1: `16/16` generation calls, `414970` tokens, `9` unique candidates;
- replicate 2: `16/16` generation calls, `430403` tokens, `7` unique candidates;
- total: `32/32` generation calls, `845373` tokens, `16` globally unique
  candidates after digest deduplication;
- discarded generation calls: `0`;
- held-out evaluations: `0`.

## BA adjudication

BlindAssist independently reevaluated every unique exported candidate with the
frozen public development evaluator and applied the preregistered ordering. The
locked public-development winner has digest
`0d04dedba8b4c8a3e9782f52b12f9e6fa615b79c584cbba1b41bf277db338c48`.

Its development result was:

- all-search-condition semantic/safety/premature hard gates: pass;
- CLEAN completion: `12/12`;
- COMBINED_MILD completion: `10/12`;
- COMBINED_MODERATE completion: `0/12`;
- COMBINED_MODERATE family completion: `0/4`, `0/4`, `0/4`;
- COMBINED_MODERATE eligible reacquisition: `1/3`;
- unsafe guidance total: `0`;
- premature completion total: `0`.

The frozen held-out admission required COMBINED_MODERATE at least `8/12`, at
least `2/4` in every family, and eligible reacquisition at least `2/3`, in
addition to the other gates. The locked winner failed those requirements, so the
encrypted held-out schedules were never released to search or used for candidate
evaluation. Their only decryption was the pre-model encryption roundtrip; the
winner-lock-gated formal opening never occurred.

The adjudication implementation was a post-run mechanical execution of the
predeclared ranking with no gate or threshold changes; its preserved SHA-256 is
`6f821b292b0f004ba3df3ca1cc7442f9cdd0372d8deb8414de77e72ac0169770`.

## Interpretation and claim ceiling

The search found policies that removed the frozen winner's public hard-gate
violations and retained clean/mild behavior, but it did not improve the primary
moderate combined condition above zero completion. Therefore GC2-B does not
establish a noise-robust search signal, robustness improvement, safety benefit,
or acceptance result.

Claim ceiling:
`symbolic_consumed_task_search_completed_no_heldout_or_robustness_claim`.

No rerun, rescue tuning, wider budget, held-out opening, real-RGB promotion, or
GC2-C successor is authorized by this closeout. A successor would need a
materially different uncertainty/state representation and a new pre-outcome
protocol rather than another search over the same six-function surface.

Authoritative evidence is under
`artifacts.local/evidence/goal-copilot/GOAL-COPILOT-2B-SKY-SEARCH/`.
