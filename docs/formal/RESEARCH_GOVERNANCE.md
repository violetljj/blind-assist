# Formal research governance

Most of this document applies only before opening protected final/blind outcomes
or publishing a claim-critical number. Ordinary reversible Development work
follows the smaller loop in `AGENTS.md`. The experiment-inheritance section
applies whenever any result is retired, reused, or compared with a successor.

## Scope of historical constraints

This scope rule also applies to ordinary Development and experiment inheritance.
Preserve original protocols, gates, outcomes, and evidence identities. A failure
constrains the tested method version, responsibility, evidence domain, and
evaluation criteria; a label or shared keyword cannot ban a mechanism family.
Before using it to exclude a future proposal, explain the actual discrepancy,
why that scope applies, and what evidence would change the judgment. Missing
evaluability is an evidence gap, never a method falsification.

A new hypothesis may change the algorithm, composition, information source,
responsibility, or evaluation criteria with a stated practical rationale and a
smallest useful check. Historical successor suggestions are revisable, not an
exhaustive route list. Choose baselines and controls for the discrepancy tested;
retrieved inheritance roles alone do not establish their applicability. Record
the new proposal separately; it needs no prior success to justify exploration.
Changing future criteria cannot retroactively pass the old experiment, erase
its failure, or restore fresh-confirmation authority to consumed evidence.

Exploration and engineering follow [the research workflow](../../research/WORKFLOW.md).
Confirmation fixes a method to test a specified claim; its required independence
depends on that claim. Enter the protected rules below for blind/final access or
claim-critical numbers, not for every Development diagnostic. A mechanical
interruption is not a method verdict; its existing retry contract still applies.

## Freeze before access

Freeze the minimum surface needed to protect the claim:

- cohort and source identities;
- implementation and model identities;
- observable inputs and evaluator-only truth boundary;
- primary metric, denominator, thresholds, and missing-data behavior;
- retry, interruption, checkpoint, and `in_doubt` semantics;
- claim ceiling and stop condition.

Record hashes for any bytes whose later change could alter the conclusion. Do
not open protected outcomes until the frozen record is internally consistent.

## Authority separation

Public goal identity, private evaluator truth, proposal, selection, and
handoff/persistence are distinct layers. Observations must not read evaluator
truth. `UNKNOWN` and `NOT_EVALUABLE` remain separate from negative outcomes.

Synthetic, replay, curated Development, pseudo-labeled, model-reviewed,
live-device, and natural-distribution evidence must be reported by their actual
source. A narrow result cannot establish a universal, product, or safety claim.

## Experiment inheritance is not a terminal status

`ACTIVE`, `GATE_NOT_MET`, `NOT_EVALUABLE`, and `CLOSED` describe the execution or
contract. They do not decide whether a method remains useful. After a terminal,
assign the exact method version a separate inheritance role for a named system
responsibility:

- `RETAINED_CORE`: the named surface still runs in the current main algorithm.
  Retain its evidence ceiling and known failure signature with the code; this is
  not a claim that every core component independently passed a gate.
- `COMPONENT_OR_CHALLENGER`: the method cannot independently advance the claim,
  but remains eligible either as a bounded `COMPONENT` inside a composition or
  as a full-system `CHALLENGER`. Record that subtype. A composition must freeze
  the interface and parent baseline, and show the component's contribution on
  new evidence; it does not inherit the component's old claim authority.
- `NEGATIVE_CONTROL`: preserve the method and its named failure signature as a
  fixed falsifier for successors. Do not weaken or retune the control merely to
  make a successor win. A clean control result detects recurrence of that error;
  it does not by itself validate the successor.
- `DEAD_FOR_THIS_ROLE`: a severe test falsified the method's core hypothesis for
  the explicitly named responsibility. Do not restore that same role through
  threshold tuning, fusion, resampling, or relabeling. Genuinely new information,
  representation, or responsibility creates a new hypothesis and must cite the
  old boundary rather than erase it.

Keep the role scoped as `method version x responsibility x evidence domain`.
The same mechanism can therefore be dead as an identity owner yet retained as a
proposal generator. `UNKNOWN` or `NOT_EVALUABLE` evidence may leave
the role unset while a result is still working material. Before it enters the
current high-authority terminal ledger, classify the surface actually inherited:
usually a bounded component or a source/contract negative control. Absence of an
evaluable algorithm test is never evidence for `DEAD_FOR_THIS_ROLE`.

Every owning current/ledger entry that may affect future work should record:

- `terminal_status` and `evidence_verdict`;
- `inheritance_role`, plus `inheritance_mode` for a component/challenger;
- `role_scope` and the exact retained surface or failure signature;
- the evidence anchor, forbidden reuse, and a decision-changing revisit trigger.

Do not backfill old roles mechanically from words such as `CLOSED`, `rejected`,
or `gate not met`. Classify an old method when it next enters an audit, baseline,
composition, or successor decision. A role change requires new versioned evidence
or an explicitly different responsibility; consumed outcomes cannot be relabeled
to upgrade authority.

The enforced current-terminal ledger is
`research/knowledge/decision/inheritance.json`. Update it through
`python tools/knowledge.py set-terminal-inheritance`; route context and diagnosis
then consume the role automatically. Only `DEAD_FOR_THIS_ROLE` blocks a same-role
candidate. Related `NEGATIVE_CONTROL` entries are required controls,
`RETAINED_CORE` entries are baselines, and components/challengers remain eligible
only in their recorded mode.

This separation follows four useful precedents: severe testing distinguishes a
failed probe from strong evidence against a specific discrepancy; negative
controls require a named null relation and do not validate the primary claim by
themselves; counterexample-guided synthesis converts failed candidates into
search constraints; and ensemble selection admits diverse weak models only when
their held-out contribution improves the combined system. See
[Mayo and Spanos](https://doi.org/10.1093/bjps/axl003),
[Penning de Vries and Groenwold](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515451/),
[Abate et al.](https://doi.org/10.1007/978-3-319-96145-3_15), and
[Caruana et al.](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).

## After access

These restrictions govern the protected final/blind run and its claim. Separate,
explicitly labeled Development diagnostics may use consumed evidence under
`AGENTS.md`; they cannot overwrite sealed outputs or regain confirmation authority.

- Do not tune, resample, fuse, or rerun a consumed arm after seeing outcomes.
- Preserve failures and partial coverage; never silently change the denominator.
- An interrupted external call is `in_doubt` unless the provider proves it was
  not consumed.
- A protected successor needs a new versioned protocol and evidence adequate
  for its claim. Algorithm, representation, or evaluation changes may define a
  new hypothesis; consumed outcomes remain Development evidence and cannot
  restore fresh-confirmation authority.
- Record both the terminal and inheritance disposition in the owning
  current/ledger, then stop when the registered condition is reached.

Use [the protocol template](RESEARCH_PROTOCOL_TEMPLATE.md) only when this formal
mode is actually entered.
