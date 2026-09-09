# Frozen LOCAL does not remedy transfer failure

2026-09-09 EXPLORE. Replacing JOINT with its existing LOCAL comparator does not
improve transfer. The 600-frame zero-fit run preserves all original B alerts and
reproduces saved B count probabilities exactly. Keep JOINT as the scoped spatial
component; do not infer that global query mixing alone explains the failures.

[Protocol](BODY_QUERY_LOCAL_TRANSFER_20260909.md),
[executed operator](body_query_local_transfer.py).

| Consumed admitted cohort | Method | Near query hit | Wrong-far | Strict pairs | Alert parity |
| --- | --- | ---: | ---: | ---: | ---: |
| Original fixture, new background |JOINT|119/159|3/53|29/53|106/106|
| Original fixture, new background |LOCAL|85/159|30/53|3/53|106/106|
| Simplified fixture, ordinary |JOINT|113/318|26/106|0/106|212/212|
| Simplified fixture, ordinary |LOCAL|67/318|64/106|0/106|212/212|
| Simplified fixture, matched size |JOINT|116/318|26/106|0/106|212/212|
| Simplified fixture, matched size |LOCAL|69/318|63/106|0/106|212/212|

The predeclared >=10-point improvement in all three arms fails. This is a fixed
architecture comparator, not a trained robustness intervention. It does not rule
out future local architectures or conclude that raw RGB contains no spatial
information. Both models share the same frozen encoder and original data support.

The source identities and exclusions are unchanged. Both original primary source
gates remain NOT_EVALUABLE; these are the previously disclosed53/106/106pair
diagnostics. Do not relabel them as fresh confirmation. Native labels enter only
saved-output scoring. No new training, source capture, threshold search, model
selection, alert replacement or ordinal extension occurred.

Checks: all input RGB, admission, truth and saved prediction hashes verified;
LOCAL checkpoint matches its original frozen hash; float64 count convolution and
Torch range threshold decisions agree; saved JOINT metrics reproduce exactly.
Original B probabilities match exactly on600frames and alerts match600/600.
Actual CUDA device RTX5060 Laptop; inference/scoring7.81seconds excluding startup.
The process is terminal; no worker or persistent compute was allocated.

Durable root: `artifacts.local/work/body-query-local-transfer-20260909/run-v1/`.
The two NPZ files and `result.json` preserve LOCAL probabilities, range outputs,
retained alerts, all hashes and this failed comparator decision.

Next input route is the [RGB plus constrained ToF observation contract](RGB_TOF_OBSERVATION_20260909.md),
following the user's linked task discussion. This is not a claim that ToF already
solves HEAD attribution; establish observation coverage and association first.
