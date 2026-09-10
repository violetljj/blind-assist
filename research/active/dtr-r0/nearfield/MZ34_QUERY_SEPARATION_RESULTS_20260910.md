# MZ34: query-separated fit not admitted by its fixed conflict gate

2026-09-10 EXPLORE, consumed TRAIN diagnostic. Execution PASS with disposition
NOT_ADMITTED_NO_FIT. MZ33's final combined shared-gradient minimum cosine is
+0.077761, above the required<=-0.1;66 responsibility errors remain. The
conjunction fails. No query-separated model was initialized, evaluated or fit.

[Protocol](MZ34_QUERY_SEPARATION_PROTOCOL_20260910.md),
[gated runner](mz34_train.py), [unfitted prototype](mz34_query_selector.py),
[gradient evidence](MZ33_QUERY_GRADIENT_RESULTS_20260910.md).

The prototype defines four separately parameterized copies of the original
selector. Its proposed initial-function and effect checks were not executed
because admission failed. It has no measured task score or performance benefit.
This outcome does not prove query sharing is harmless or independent query
parameters ineffective. It closes this conflict-gated fit without weakening
the rule after the diagnostic. A different rationale requires a separate brief.

The runner verified the bound MZ30/MZ32 and MZ31/MZ33 receipts, then saved only
admission.json and receipt.json in
artifacts.local/work/mz34-query-separation-20260910/run-v1/.
No checkpoint, optimizer, task prediction or training loss was produced.
The absence of fit artifacts and exact gate recomputation were checked; source
syntax checks cover the prepared prototype without claiming model execution.
Retain the admission outcome as a diagnostic COMPONENT, not a fitted challenger.
No model, cutoff, App/default, source, temporal or hardware state changed.
