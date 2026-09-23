# Exact first-contact supervision at fixed representation

User-authorized EXPLORE following the direct-CDF negative control. Test whether
adding explicit position/censoring constraints improves train fit and held-layout
boundaries with the SAME MetricContactModel. This changes supervision information
and objective, not the model. It is not a rescue of the old binary-only experiment.

One new100epoch fit,864train/288dev/576evaluation images, same consumed48layouts,
sampled72queries pertrain image, frozen4864D RGB/ToF features, train normalization,
saved direct-CDF initial weights and100epoch image order, batch32,AdamW1e-3/decay1e-4.
The saved binary-only selected CDF checkpoint is the primary paired control;
sampled geometry remains an additional stronger reference. No control retraining.

For each original sampled width/layer, use TRAIN geometry only to identify first
axial contact in[.3,3]m regardless of the query horizon. Finite contact gets its
exact coordinate; beyond3m/no contact gets +inf, meaning right-censored at3m.
Train objective = inherited positive-weight50168/12040 binary BCE + weight1
censored interval NLL. A finite interior z supervises the CDF probability inside
[z-.025,z+.025] clipped to[.3,3]. At z=.3, supervise all modeled mass through.325;
at no contact, use -log(1-q). Exact3m contact remains finite, not censored.
Stable analytic logistic differences are unit-tested. The added labels reveal
more than original binary answers, including beyond each sampled horizon. This is
intentional privileged TRAIN supervision, never observation or an inference input.
Repeated width/layer queries retain their original multiplicity; same72 exposures.

Same dev original72binary-query BCE every10epochs chooses checkpoint and inherited
max recall atdevFPR<=5% chooses cutoff. No dev exact metric labels enter selection.
Record each checkpoint's train loss components and train metric summaries (diagnostic
only) to distinguish fitting trajectory from selected performance. These diagnostics
cannot change epoch/cutoff. Freeze selected train/dev/evaluation predictions before
joining evaluation targets. Loading a mixed consumed geometry JSON then indexing
train rows is code-path separation, not process isolation or fresh confirmation.

Primary supervision effect: selected horizon within5cm +10points versus binary-only
CDF on BOTH train and evaluation; false horizon crossing rate rise<=2points,
query recall loss<=3points,FPR rise<=2points,width hit loss<=3points on evaluation.
Report every term and partial gains; also compare against sampled geometry without
selecting a new winner/threshold. Preserve full finite-truth denominator, missing,
conditional MAE/p90, right-censored falsecross, subgroups, pair ranking, UNKNOWN,
width/horizon reversals and original full component gate. Saved metadata UNKNOWN
does not mean validated abstention. Train trajectory diagnostic atcutoff.5 is fixed
and cannot replace selected dev-cutoff results. Also report continuous crossings.

If train and transfer improve, retain supervision as a component with actual costs.
Train-only gain suggests transfer limitation. Neither improving leaves this fixed
representation/parameterization/optimization unresolved, not proven sensor failure.
One seed/generator; no epochs,loss weights,bins,cutoffs,seeds or automatic frontier
successor. End after fit, independent train-label and saved-output audit, documented
disposition and scoped Git delivery. A/LOCAL/App unchanged. No hardware/safety claim.
