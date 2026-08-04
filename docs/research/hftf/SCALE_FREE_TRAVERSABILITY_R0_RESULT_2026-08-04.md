# Scale-Free Traversability R0 Result

Date: 2026-08-04

Protocol terminal:
`SCALE_FREE_TRAVERSABILITY_R0_EXECUTES_STABLY_DEVELOPMENT_ONLY`

Integration decision: `KEEP_OFFLINE_DIAGNOSTIC_ONLY_DO_NOT_INTEGRATE`

The frozen scale-free operator ran on all 75 frames from three consumed phone
sessions. It read neither the corrected camera height nor Samsung Quick Measure
distance. All three sessions reached 100% score execution coverage and exceeded
the frozen 0.80 modal-label stability gate, so the mechanics terminal passes.
This is stability, not accuracy.

## Session result

| Session | Score execution | Post-warmup output | Non-ambiguous coverage | Modal fraction | Median DA latency |
|---|---:|---|---:|---:|---:|
| `DEV-20260804-125252` | 25/25 | 20 ambiguous, 1 center | 4.76% | 95.24% | 53.06 ms |
| `DEV-20260804-130313` | 25/25 | 21 center | 100% | 100% | 53.47 ms |
| `DEV-20260804-130930` | 25/25 | 21 left | 100% | 100% | 54.45 ms |

The first scene demonstrates useful abstention rather than a forced direction.
The other two scenes are temporally stable, but there is no independent
traversability truth and the camera did not move. Visual QA confirmed portrait
orientation, band rendering, and the absence of metric/safety wording; it also
confirmed that a stable relative label cannot be visually promoted to a correct
route decision. No percentile, band, margin, or temporal rule was changed after
the outputs were read.

## Authority boundary

- The label means only “lowest row-relative intrusion score among these three
  image bands.” It does not mean clear, safe, blocked, a distance, collision
  probability, or future prediction.
- These fixed-camera clips cannot evaluate approach motion, user motion,
  obstacle passage, low light, blur, stairs, or outdoor transfer.
- The three sessions are independent units; 75 frames are not treated as 75
  independent examples.
- The result authorizes retaining an offline diagnostic and designing an
  independent evaluation later. It does not authorize App integration, alerts,
  default-route changes, safety, or production.

Durable result SHA-256:
`ED1584350A171A73DECB819D16B7D167413ED87784F11C0643A28944BE5361B3`.
Frame ledger SHA-256:
`794E2F2CB18BBFAEEC43F1F06262ADB07A43FB13E67A77DCD252709ABD734AC8`.
Full JSONL and three annotated MP4 files are retained under
`artifacts.local/evidence/hftf/scale-free-traversability-r0-phone-consumed-20260804/`.
