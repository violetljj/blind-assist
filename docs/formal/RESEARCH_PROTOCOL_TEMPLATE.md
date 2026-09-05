# Formal protocol template

Use this template before protected outcome access. Delete sections that are not
material; do not add ceremony without a named risk.

## Identity

- Protocol ID and date:
- Practical capability, question, and primary claim:
- Explanatory hypothesis and comparison with incumbent/simple baseline:
- Cohort/source identity and hashes:
- Implementation/model identity and hashes:
- Evaluator identity:

## Frozen evaluation

- Observable inputs:
- Evaluator-only truth:
- Primary metric and denominator:
- Thresholds and missing-data behavior:
- Baseline and comparison:
- Stop condition and decisions for gain, no gain, or not evaluable:
- Claim ceiling:

## Execution semantics

- Budget and attempt count:
- Retryable versus terminal failures, distinguishing engineering from method evidence:
- Checkpoint/resume rule and evidence proving safe continuation of this frozen run:
- Task-owned resource release on completion, failure, or interruption:
- `in_doubt` accounting:
- Output and receipt paths:

## Result

- Terminal status:
- Primary metric:
- Coverage and exclusions:
- Failure mechanism:
- Supported claim:
- Unsupported claims:
- Next decision or closure:
