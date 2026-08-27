# Host research compute

This route covers local or remote training, offline evaluation, and other long
research jobs. It does not define Android/device execution and contains no
machine-specific paths or hardware assumptions.

## Current DTR-R0 entrypoint

```powershell
pwsh -NoProfile -File tools/ba.ps1 doctor research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 smoke research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 run research-dtr-r0 -EventInput <events.jsonl> -ResultOutput <result.json>
```

Resolve Python and output paths through the `research-dtr-r0` profile. Keep
event ledgers, videos, model outputs, logs, progress state, and results under
ignored `artifacts.local/`.

## Backend and throughput

- Use a short representative benchmark before choosing batch size, worker
  count, or backend.
- When scientific results are equivalent, use the measured faster path.
- Prefer GPU for suitable batched tensor workloads; do not move archive I/O,
  Python control flow, or tiny operations to GPU merely to use it.
- Runtime speed is execution evidence, never algorithmic uplift.
- Preserve scientific batch, seed, threshold, metric, and denominator while
  benchmarking execution alternatives.

## Long-job minimum

A job likely to outlive the current interactive window needs:

- an explicit output directory and one owning process;
- bounded startup validation on representative input;
- machine-readable progress when completed/total is observable;
- a checkpoint or an honest statement that partial work cannot resume;
- a terminal success or failure artifact;
- cleanup instructions for task-owned processes and temporary resources.

Progress should report completed/total, current stage, last activity, failures,
and an evidence-based ETA or `unknown`. Monitoring is read-only and must not
change inputs, checkpoints, outputs, budgets, or evaluator state.

## Recovery

Resume only after confirming input/config identity, output ownership, checkpoint
integrity, and that no second writer is active. Never reset a frozen budget or
silently repeat an `in_doubt` external call. If iteration-level resume is not
implemented, state the maximum lost work before launching the long job.

## Escalation

Ordinary reversible Development training remains `EXPLORE`; duration alone does
not create a formal protocol. Enter [formal research governance](formal/RESEARCH_GOVERNANCE.md)
only before protected final/blind outcome access or a claim-critical one-shot
run. External paid compute additionally requires explicit resource ownership,
budget, lifecycle, and cleanup boundaries.
