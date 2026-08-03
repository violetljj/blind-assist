# Known-Camera-Height Ground Scale: Consumed Development Result

Date: 2026-08-04

Decision: `KEEP_AS_NON_PROMOTABLE_DEVELOPMENT_SIGNAL_STOP_THIS_OPTIMIZATION_BRANCH`

The user explicitly allowed consumed data for development. We therefore ran the frozen R0 operator on five height-eligible TartanGround parents and 165 frozen anchors. This is consumed synthetic Development evidence only; it is not fresh, held-out, real-phone, product, or generalization evidence.

## Result

| Parent-macro metric | Raw DA | Known height R0 | Causal median R1 (9 valid scales) |
|---|---:|---:|---:|
| known coverage | 0.6485 | 0.6364 | 0.7212 |
| clearance MAE (m) | 1.0424 | 0.4710 | 0.3949 |
| envelope agreement | 0.5328 | 0.8168 | 0.8475 |
| false-clear rate | 0.4672 | 0.1814 | 0.1510 |
| temporal-delta MAE (m) | 0.5335 | 0.3101 | 0.2676 |

R0 materially improved four error metrics and was jointly better than raw DA on 3/5 parents. Its scale-free plane recovery nevertheless selected the wrong effective scale often enough that clearance MAE, agreement, false-clear, and temporal gates all failed.

The bounded posthoc comparison tested causal median windows 1, 3, 5, 9, 15, and 33 on the same consumed outcomes. Window 9 improved every R0 parent-macro metric shown above, uses no future frame, and was frozen in the [R1 optimization receipt](KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R1_POSTHOC_OPTIMIZATION_RECEIPT_2026-08-04.md). It still failed the same four absolute gates. The terminal is `POSTHOC_CONSUMED_R1_ABSOLUTE_GATES_FAIL_STOP_OPTIMIZATION`; no further threshold, window, model, or outcome-conditioned selector search is authorized on this branch.

## Source and evidence boundaries

The fresh ARKit attempt remains separately held: only 2/4 locked parents passed the pre-DA height-proxy source gate, so DA and effect outputs stayed unread for that cohort. The existing ARCore SM-S9280 route also remains `NOT_EVALUABLE` because it did not produce exact-timestamp raw depth. Neither route justifies buying ToF hardware or changing the default app.

Durable R0 result SHA-256: `CF32CF6F57504AF51633AC97F7DCC74C97B37C0F559181A6463051E8E5B6F0BE`.

Durable R1 result SHA-256: `DB841CE3344F06F6F42CF55622EDB1C8E8F0CE4FB96504C235EA31469BDB0834`.

The full frame records and cached DA predictions remain under ignored `artifacts.local/evidence/hftf/known-camera-height-ground-scale-consumed-development-20260804-run2/`.
