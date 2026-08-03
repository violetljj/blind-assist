# Corridor-conditioned future field A2.1 fresh result

Date: 2026-08-03

Terminal: `CORRIDOR_CONDITIONED_FUTURE_FIELD_A2_1_FRESH_NOT_SUPPORTED`

The frozen A2.1 model failed three of six gates on seven unopened TUM
`sitting_halfsphere` windows and 1,338 known future opportunities. No model,
threshold, feature, window, reference value, or gate changed after source open.

| Measure | Frozen field | Best fixed comparator | Gate | Pass |
|---|---:|---:|---:|:---:|
| Brier | 0.10149 | HOLD 0.09712 | >=15% reduction | no (-4.50%) |
| Log loss | 0.33060 | HOLD 0.33248 | >=20% reduction | no (+0.57%) |
| ECE | 0.08170 | - | <=0.10 | yes |
| Recall | 82.33% | - | >=85% | no |
| FPR | 8.91% | - | <=15% | yes |
| MCC | 0.68488 | HOLD 0.67793 | strictly best | yes |

The model became conservative under the walking-to-sitting regime change: it
reduced false positives and retained the best MCC by a small margin, but missed
too many future occupied cases and no longer improved proper scoring rules over
HOLD. This does not support a transferable 0.5-second future occupancy claim.

The result does not invalidate the separately frozen A0.1 current-occupancy
fresh result. It narrows the branch: current collision probability is supported
within TUM Development evidence; explicit future occupancy remains unsupported.
No A2.1 retraining, recalibration, or threshold successor is allowed on this
source.

## Source identity

- Archive bytes: `651422497`
- Archive SHA-256:
  `BA9F0FAB0D07E22F04FBFAE16EB4E3FB44088A32C920AD36C782B5024ED4B767`
- Extracted RGB/depth frames: 1,110 / 1,082
- Manifest SHA-256:
  `52E9E0D62CE5A222CDADF7C5B9C5B3DBEBD00B6740887C39D126C72C03800460`
- UniDepth current-field report SHA-256:
  `70BB931860C4F5149AD5672F2C9C002150A342BE48A4272516CC82A38707AB7B`
- Frozen future-field report SHA-256:
  `770A40AB12C3FC72E405D58DC99784A81D7E3A102FB678353A134445402EB8A4`
