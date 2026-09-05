# Formal research governance

This document applies only before opening protected final/blind outcomes or
publishing a claim-critical number. Ordinary reversible Development work follows
the smaller loop in `AGENTS.md`.

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
- Record the terminal in the owning current/ledger and stop when the registered
  condition is reached.

Use [the protocol template](RESEARCH_PROTOCOL_TEMPLATE.md) only when this formal
mode is actually entered.
