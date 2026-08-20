# L10M-B2: Seed-89 Candidate Transplant

B2 resolves the only localized ambiguity left by the evaluable B1 successor. It
does not repeat search or add seeds. It freezes the seed-89 Raw generation-2
`action_selection` intervention, deterministically renders that same
`PolicySpec` through the Raw and Structured interfaces, parses both, and sends
both to the same frozen B1 evaluator.

The claim is deliberately narrow: equality can localize the seed-89 difference
to search trajectory inside this finite synthetic interface, but cannot establish
general Structured-search value, end-to-end behavior, device behavior, user
benefit, or safety effects.

Freeze before evaluating:

```text
python -m scripts.research.l10m_b2.candidate_transplant freeze \
  --b1-run-dir artifacts.local/evidence/l10m_b1/runs/b1-20260820T115002-98733875 \
  --output artifacts.local/evidence/l10m_b2/seed89_candidate_transplant/protocol.json
```

Then run the deterministic transplant with zero search/model calls:

```text
python -m scripts.research.l10m_b2.candidate_transplant run \
  --b1-run-dir artifacts.local/evidence/l10m_b1/runs/b1-20260820T115002-98733875 \
  --protocol artifacts.local/evidence/l10m_b2/seed89_candidate_transplant/protocol.json \
  --output artifacts.local/evidence/l10m_b2/seed89_candidate_transplant/result.json
```

## Terminal

`B2_EVALUABLE_COMPLETE / B2_SEARCH_PATH_FAILURE_SIGNAL`

The frozen intervention compiled to the same canonical spec through both
interfaces (`b0110121...5021`), and the complete evaluator outputs shared the
same behavior hash (`ba2eb95f...7919`) and score (`0.993103448275862`). Thus the
seed-89 B1 difference is localized to the observed search trajectory, not a
Structured expressivity failure for this intervention inside the frozen finite
interface. This does not establish general Structured-search value.

The create-once local receipts are bound as:

- `protocol.json`: `15d0cbf4d7cdd1d89440a2967eb7fd0b74b5caf2d924cad4e1fbe86052869926`
- `result.json`: `7fe39bac5aa31d049dc95628c392b56789c676a2815b06fe6f9bc69df72df9e7`
