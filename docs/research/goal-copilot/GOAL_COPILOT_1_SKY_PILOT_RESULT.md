# GOAL-COPILOT-1-SKY-PILOT result

Final verdict:

`GOAL_COPILOT_1_SKY_SEARCH_SIGNAL_ESTABLISHED_ON_SEALED_PILOT`

SkyDiscover's canonical `best_of_n` search produced a locked policy that passed
the preregistered deterministic Pilot gate. This establishes a search signal on
this small symbolic closed-loop Pilot only. It does not establish real-vision
performance, user safety/effectiveness, population-level statistics, or general
superiority.

## Primary results

| Development (12) | Baseline | Locked winner |
|---|---:|---:|
| Completion | 6/12 | 12/12 |
| Find & Reach | 2/4 | 4/4 |
| Track/Reacquire | 2/4 | 4/4 |
| Align/Interact | 2/4 | 4/4 |
| Unsafe guidance | 0 | 0 |
| Premature completion | 0 | 0 |

Both independent replicates produced a hard-gate-valid, baseline-beating dev
candidate. The frozen dev selection locked candidate
`24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1`
from replicate 1 before fresh was opened.

| Sealed fresh (6) | Baseline | Locked winner |
|---|---:|---:|
| Completion | 3/6 | 6/6 |
| Find & Reach | 1/2 | 2/2 |
| Track/Reacquire | 1/2 | 2/2 |
| Align/Interact | 1/2 | 2/2 |
| Unsafe guidance | 0 | 0 |
| Premature completion | 0 | 0 |
| Baseline-completed regressions | — | 0 |

Fresh completion improved by +3, with +1 in every family. The winner retained
every baseline completion and completed at least one scenario in every family;
all preregistered fresh PASS gates therefore passed.

## Search execution

- Frozen implementation commits: BlindAssist
  `498081ee2e5813fee8b49a9a346fb0d4b60309da`; SkyDiscover
  `837f3243b37b361e927785d2a9e36777fc17802a`.
- Provider: native `E:\codex-tools\bin\codex.exe`, `codex-cli 0.148.0`,
  executable SHA-256 `2ad2cf8a732da68b8f141634f92db1a03016c5faf533a7225fbc0fb740130410`,
  ChatGPT authentication, `gpt-5.6-sol`, reasoning `medium`, Responses wire API.
- Replicate 1: 16 generation attempts, 16 evaluations, 16 candidates/14 unique,
  391,193 tokens, COMPLETE.
- Replicate 2: 16 generation attempts, 16 evaluations, 12 candidates/12 unique,
  396,687 tokens, COMPLETE.
- Total: exactly 32 generation attempts and 32 evaluations; 28 evaluable
  candidates, 26 unique; four failed candidate attempts were not exported as
  eligible candidates. No retry, resume, replacement, or in-doubt call occurred.
- Replay recomputed baseline/winner dev and fresh assessments exactly, verified
  the winner and seal hashes, confirmed both journals complete, and found no
  fresh scenario identifier in the SearchTaskBundle or search outputs.

## Required A–T closeout

- A. protocol seal digest:
  `1ad3284b4d359f43605515b924f7653f84e99c9cd7b83d88db69dd49d6f40414`
- B. SearchTaskBundle digest:
  `063228bec20cbd74adb724f996133f5a26657fe0be5af04b4c3360cee59dedec`
- C. sealed fresh cohort digest:
  `2628d025a628aeb03281688bed4f8a7ca19f0ffaa4b1991635f5610c94115293`
- D. BA frozen execution commit: `498081ee2e5813fee8b49a9a346fb0d4b60309da`
- E. Sky frozen execution commit: `837f3243b37b361e927785d2a9e36777fc17802a`
- F. provider/model identity: qualified native Codex CLI 0.148.0 / ChatGPT /
  `gpt-5.6-sol` / medium / Responses.
- G. attempts/candidates/tokens: 32 generation, 32 evaluation, 28 candidates,
  26 unique, 748,710 input + 39,170 output = 787,880 total tokens.
- H. replicate 1: COMPLETE; 16/16; 16 candidates/14 unique; dev best 12/12.
- I. replicate 2: COMPLETE; 16/16; 12 candidates/12 unique; dev best 12/12.
- J. baseline dev completion: 6/12.
- K. winner dev completion: 12/12.
- L. `fresh_execution_authorized=true`.
- M. baseline fresh completion: 3/6.
- N. winner fresh completion: 6/6.
- O. family gains/losses: fresh +1/+1/+1; no completion losses.
- P. winner unsafe count: 0 dev, 0 fresh.
- Q. winner premature completion count: 0 dev, 0 fresh.
- R. replay status: PASS.
- S. exact final verdict:
  `GOAL_COPILOT_1_SKY_SEARCH_SIGNAL_ESTABLISHED_ON_SEALED_PILOT`.
- T. `resume_authorized=false`.

Machine-readable authority is the ignored local evidence root
`artifacts.local/evidence/goal-copilot/GOAL-COPILOT-1-SKY-PILOT/`, especially
`formal_protocol_seal.json`, `adjudication/winner_selection.json`,
`adjudication/fresh_analysis.json`, `replay/replay_receipt.json`, and
`formal_closeout.json`.
