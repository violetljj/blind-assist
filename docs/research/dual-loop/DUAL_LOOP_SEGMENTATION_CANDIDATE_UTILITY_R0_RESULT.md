# DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0 formal result

## 结论

正式 blind holdout 已完成，唯一终态为：

**CURRENT_SEGMENTATION_REFERENCE_REJECTED**

这不是对 segmentation 类别本身的全面否定，而是对当前固定 reference、当前
三臂 fusion operator 和预声明成本/误激活门的 Development 关闭。结果不授权
Android、QNN、风险事件、主动提醒、产品安全或真实世界可通行性结论。

## 固定执行

| 项目 | 值 |
| --- | --- |
| protocol | DUAL_LOOP_SEGMENTATION_CANDIDATE_UTILITY_R0 |
| protocol SHA256 | 8036e22a6e03b94efa17329a81e7902ca39b0ebf59c5bb4b2895ff29d840c632 |
| pixel truth | SANPO-Real v0 canonical R3, source_ground_truth |
| formal manifest | blind_holdout/manifest.jsonl, 120 frames, 2 source sessions |
| manifest SHA256 | b6ed5735ee3b26018c251613085fa910355f45397f8b010f6fadfd95eecba6da |
| segmentation reference | sanpo-v3-pretrained-weighted-best-int8-20260713.tflite |
| model SHA256 | 88f0184d2671230c1f1f43192758689d286b530d7490e1d1ca0671f83b50b50c |
| YOLO trace | host reference trace, 120 exact identity pairs |
| YOLO trace SHA256 | 25701f07182b615cf458ddb3d4d5af9583bdfd9e3c5479e8b197100de631af54 |
| analysis grid | 256 x 256 |
| hazard | obstacle union boundary_step_curb |
| unknown_nonwalkable | separate ablation only |

三臂严格保持为 YOLO-only A、segmentation-only B、YOLO + segmentation C；
candidate 是 H_t minus D_t，truth counterpart 是 source-native hazard minus D_t。

## Formal metrics

| 指标 | blind formal value | frozen gate | 结果 |
| --- | ---: | ---: | --- |
| C minus A pixel recall | 0.073670 | >= 0.05 | pass |
| C minus A false-positive area | 0.039236 | <= 0.05 | pass |
| candidate component recall | 0.688129 | >= 0.50 | pass |
| false activation components/frame | 13.833333 | <= 3.0 | **fail** |
| consistent source sessions | 2 | >= 2 | pass |
| segmentation host P95 | 24.621945 ms | <= 25 ms | pass |
| total incremental host P95 | 138.443840 ms | <= 30 ms | **fail** |

source-wise C minus A recall 为：

- center_obstacle：0.394242；candidate component recall 0.675462；
- step_curb：0.032040；candidate component recall 0.728814。

因此像素增量和 component recall 在两 session 方向上都为正，但当前 candidate
产生的 false activation 过多，且整段增量成本显著超过冻结门。calibration
dev split 也提前显示同一成本风险；formal 只用于终态确认，没有反向调参。

## Temporal and runtime boundary

formal report 同时输出 raw adjacent IoU、motion-warped IoU 字段、birth/death、
persistence、split/merge、flicker 和 runtime percentiles。当前 formal 没有可信
motion sidecar，因此 motion_warp_available=false，未用 identity warp 冒充
补偿；raw temporal diagnostics 可用，motion-warped 值保持 not available。
protocol 允许在 pixel utility terminal 中不把缺失 motion warp 当作额外通过门，
但这不构成自然视频 temporal transfer 结论。

## 验证与证据

- evaluator report：
  artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/report.json
- frame rows：
  artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/frames.jsonl
- component ledger：
  artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/components.jsonl
- independent validation：
  artifacts.local/evidence/dual-loop-segmentation-candidate-utility-r0/formal/validation.json
- validation SHA256：
  7dacae095168d9d47fe10bf91888138ce8a00f6fe8e9496c3d26ac23a58b512b

独立 validator 复算了 frame confusion、aggregate pixel metrics、candidate
component aggregate、packed mask 长度、身份唯一性、runtime 非负性和 temporal
字段完整性；validation_status=VALID。

## 后续边界

当前 segmentation reference 关闭。后续若要继续，必须另行冻结新的 reference
或更小的 fusion operator，并重新声明 calibration、false-activation、runtime
和 motion-input 资格；本结果不授权添加 Agent labels、扩大 unknown_nonwalkable、
叠加复杂状态机或接入主动提醒。
