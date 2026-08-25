# Formal research governance

This document applies only before opening protected final/blind outcomes or
publishing a claim-critical number. Ordinary reversible Development work follows
the smaller loop in `AGENTS.md`.

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

- Do not tune, resample, fuse, or rerun a consumed arm after seeing outcomes.
- Preserve failures and partial coverage; never silently change the denominator.
- An interrupted external call is `in_doubt` unless the provider proves it was
  not consumed.
- A successor needs a versioned protocol and a genuinely new information source,
  not threshold or backbone rescue of the consumed route.
- Record the terminal in the owning current/ledger and stop when the registered
  condition is reached.

Use [the protocol template](RESEARCH_PROTOCOL_TEMPLATE.md) only when this formal
mode is actually entered.
