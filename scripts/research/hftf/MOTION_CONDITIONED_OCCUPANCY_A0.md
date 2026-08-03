# Motion-conditioned occupancy A0

Date: 2026-08-03

Status: `FROZEN_BEFORE_WINDOW_LOSO_OUTCOME`

## Question

Can HFTF-style causal camera-motion evidence make a low-capacity collision
occupancy probability transfer across whole windows, where depth-only and
confidence-stratified post-processing failed?

## Fixed data and isolation

- Inputs: fixed-world-floor UniDepth reports for the four consumed
  `walking_static/xyz` windows and six consumed `walking_rpy` windows.
- Evaluation: leave one complete 3-second window out; all band×horizon rows
  from that window are held out together.
- Labels: registered-depth fixed-world-floor occupancy, read only by the fit and
  evaluator for the appropriate folds.
- No fresh source is opened for this Development screen.

## Fixed features

Per band×horizon row:

- predicted clearance minus horizon;
- predicted clearance and horizon;
- UniDepth clearance-support `log1p` confidence;
- ground-plane median residual;
- `log1p` obstacle point count;
- left and centre indicators (right is reference);
- causal RAFT-small previous-to-current global median/P90 flow magnitude;
- normalized affine translation x/y, absolute rotation and log scale;
- affine inlier fraction;
- residual-flow median/P90 magnitude;
- first-frame motion-missing indicator.

RAFT-small uses the already frozen official checkpoint SHA-256
`01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`
at `224x128`. The classifier is standardized unweighted Logistic Regression,
fitted deterministically with L2 coefficient `0.01`; no polynomial features,
class weights, threshold search, or hyperparameter selection are allowed.

## Continuation gates

The same six probability gates remain:

1. Brier reduction `>=15%` versus deterministic clearance;
2. log-loss reduction `>=20%`;
3. ECE `<=0.10`;
4. false-clear among `P(occupied)<=0.05` outputs `<=5%`;
5. such high-confidence-clear coverage `>=10%`;
6. occupied recall at `P(occupied)>=0.50` `>=85%`.

All must pass in pooled window-LOSO predictions before a final model may be
frozen for `walking_halfsphere`.
