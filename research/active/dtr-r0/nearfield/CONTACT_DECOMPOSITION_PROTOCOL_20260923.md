# Frozen contact-probability / position / width diagnostic

User-authorized read-only EXPLORE. No training, checkpoint/threshold replacement,
architecture change or App action. Compare only frozen selected binary-CDF(epoch50)
and exact-supervision CDF(epoch10). The later exact-training curves do not supply
checkpoints and are not promoted. Preserve old terminal dispositions.

Extract q,mu,scale on shared widths.2..1.2m at.02mspacing, BODY/HEAD, H=3, for
864train and576layout-held consumed images. Use original features/normalization,
checkpoints and their dev-selected cutoffs. Frozen-protocol CPU-only replay matches
both original selected prediction backends; record torch backend. Seal component
arrays before joining geometry and truth. Recompute original q atH3 and width.6
continuous crossing against frozen predictions, reporting numerical differences.

Primary focus is width.6. Truth partitions remain all finite interior first-contact
positions, left-censored, and right-censored at3m; no-contact positions are not zeros.
Conditional median solves F(z)/F(3)=.5 (clipped at.3). It always has a location and
is NOT an unconditional obstacle prediction. Report all-finite5cm hit/MAE/p90,
including samples blocked by q<cutoff. Partition finite truths into q-admitted/blocked
x conditional-median accurate/inaccurate; additionally show admitted median-good
but actual-crossing-bad and the reverse. Actual crossing solves G(z)=cutoff/q.
The q=1 same-cutoff intervention is diagnostic only and MUST report right-censored
false crossings; it cannot justify lowering thresholds or declaring clear space.
Report raw mu separately, scale and conditional10-90%span on fixed partitions.
Narrow span<=.1m with median error>.05m is descriptive, not calibrated uncertainty.

Compare train and evaluation distributions without claiming memorization is proven:
different layouts and truth composition can also change performance. All-width
summaries retain correlated image/layer/width counts, not independent trials.
For each curve count q decrease>1e-6 over ALL adjacent widths; conditional-median
Z increase>1cm on pairs where BOTH true contacts are finite; admitted-contact
disappearance and actual finite-crossing increases>1cm. Audit truth monotonicity.
Report anchor BODY/HEAD separately and keep original sensor UNKNOWN metadata.

No gate or hypothesis selection from a new cutoff. Describe the quantitative share
of blocked cases with accurate internal position, inaccurate positions even when
admitted, transfer gaps and physical-order violations. These diagnostics cannot
uniquely identify causal source because q/location/scale are jointly learned.
An apparent useful position-only signal is a follow-up hypothesis, not an algorithm.

Use existing consumed geometry after component seal; independent world-coordinate
target and numerical decomposition audit. No new capture,paid allocation or services.
End after one extraction percheckpoint, arithmetic/tests/audit, result/inheritance,
registry and scoped Git delivery. Retain all payloads in canonical artifacts.local.
