# Semantic Distinctive Anchor V1 Result

状态：`CONTROLLED_DEVELOPMENT / SAME_SEQUENCE_EVIDENCE_INTERVENTION / PASSIVE_TOP1_11_OF_16_TO_SEMANTIC_16_OF_16 / WRONG_LOCK_9_TO_0 / REACQUISITION_3_OF_4_TO_4_OF_4 / SEMANTIC_INFORMATION_GAIN_DEMO_PASS / PASSIVE_APPEARANCE_MAINLINE_REMAINS_CLOSED / ANDROID_MARKER_POSE_IMPLEMENTED_LIVE_DEVICE_NOT_RUN / DEFAULT_APP_UNCHANGED`

## 结论

`SEMANTIC_DISTINCTIVE_ANCHOR_V1` 已把上一轮缺失的 OCR runtime、三种独立 anchor、受控 demo 与结果表一次性打通。
它复用 Active Distinctive V0 完全相同的 4 targets、16 个 target-present decisions、candidate slot/role 与 4 次
lost/reacquisition 节奏；冻结 passive receipt 不重跑。唯一变化是主动观察现在可取得 goal-selected semantic evidence，
且 semantic arm 在 anchor 不唯一或不存在时只 `ABSTAIN`，不回退 appearance，也不给 tracker identity authority。

| 指标 | frozen passive appearance | semantic anchor V1 | delta |
|---|---:|---:|---:|
| target top-1 | 11/16 | **16/16** | **+5** |
| wrong-target lock（20 个 sequence steps） | 9 | **0** | **-9** |
| lost 后 fresh reacquisition | 3/4 | **4/4** | **+1** |
| target-lost abstention | 0/4 | **4/4** | **+4** |

这建立的是受控 mechanism：当观察中出现真正独立、可解码的身份或 goal semantics 时，active acquisition 才产生
appearance-only 没有的信息增益。它不重新打开 passive exact-instance backbone/patch/layout 路线。

## 数据与三类 anchor

| modality | target-present | passive | semantic | lost 行为 | 证据性质 |
|---|---:|---:|---:|---|---|
| natural OCR `COFFEE` | 4 | 4/4 | 4/4 | abstain | 既有 Wikimedia Starbucks frame 中自然出现；无 overlay |
| distinctive sign patch `HOUSE BAKE` | 4 | 0/4 | 4/4 | abstain | 从公开 reference 固定裁出的目标语义 sign，作为 active-scan 可见干预 |
| OCR product code `BA101` vs `BA102` | 4 | 3/4 | 4/4 | abstain | Washington 商品帧上的确定性 printed-code canary |
| ArUco `17` vs `23` | 4 | 4/4 | 4/4 | abstain | Washington personal-item 帧上的 DICT_4X4_50 marker canary |

除 natural OCR 外，其余三项都是明确披露的 deterministic derived canary，不是现场自然分布。Marker/唯一文字码可在
本 demo 内直接承担 identity；logo/sign patch 只承担当前 goal 的 scoped semantic authority，现实中若重复出现不能证明
physical instance。这里比较的是“旧 observation 信息”与“加入独立 anchor 后的新 observation”，不是 same-pixel matcher
A/B，也不是 general open-world OCR/logo benchmark。

## Runtime、算法与 demo

OCR runtime 隔离在 ignored `artifacts.local/runtime/semantic-anchor-v1/`：RapidOCR `3.9.2`、实际导入
ONNX Runtime `1.26.0`、OpenCV `4.10.0`。PP-OCRv6 det/rec 与 classifier ONNX 均在 raw receipt 中记录绝对路径、
bytes 与 SHA-256。安装脚本不会修改共享 Python 或默认 App。

决策规则只有一个：每个 candidate 独立解码预期 OCR substring、ArUco ID 或经 homography 验证的 fixed sign template；
恰好一个 candidate 命中才 `LOCK`，零个或多个命中都 `ABSTAIN`。没有 DINO fallback、score fusion、belief、tracker、
margin/threshold sweep 或 outcome 后 rescue。

运行命令：

```powershell
pwsh -NoProfile -File `
  scripts/research/goal_copilot_bridge/public_identifiable_referent_contract_v1/install_semantic_anchor_v1_runtime.ps1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.semantic_distinctive_anchor_v1 `
  --source-run-dir artifacts.local/evidence/active-distinctive-evidence-acquisition-v0/run-20260824T100000Z-r2 `
  --runtime-root artifacts.local/runtime/semantic-anchor-v1 `
  --run-dir artifacts.local/evidence/semantic-distinctive-anchor-v1/run-20260824T180000+0800-r2
```

最终 evidence：`artifacts.local/evidence/semantic-distinctive-anchor-v1/run-20260824T180000+0800-r2/`。
R1 只暴露并定位了共享 ONNX Runtime import 与 metadata 版本不一致；R2 把 `onnxruntime==1.26.0` 隔离固定后，未改数据、
anchor、阈值或决策，指标逐项相同。

| 文件 | SHA-256 |
|---|---|
| `cohort-manifest.json` | `a8d38426f7e8c9a447a7c475bd89b0a1702454cf6866f2b6fd17b78059f473aa` |
| `raw-decisions.json` | `f6204ab5582049517a4934b35085454d2275609c5640fce257d3231803cdcb17` |
| `final-report.json` | `0ef88603d4055665d7301f7270cf869dbf120f8ad7420092db336712abe7d18f` |
| `demo-board.jpg` | `de483f5cac567176e300b55e4eca523d24ce12952d872f091ef397123b8d8f8d` |

## 下一边界

本结果满足“先看到明显 uplift”再接完整闭环的前提。下一实现应是 research-only 的现场
`SEARCH -> SEMANTIC LOCK -> LOST -> FRESH REACQUIRE` seam：marker 先做设备 canary，OCR 进入自然门牌/店名，logo/sign
只在目标语义足够 unique 时锁定；短时 tracker 只能维持已经由 semantic anchor 确认的连续性，不能自行创建或恢复 identity。

后续 [`SAGE-LM V2-MARKER-POSE live seam`](SAGE_LM_V2_MARKER_POSE_LIVE_SEAM_IMPLEMENTATION_2026-08-25.md)
已在独立 Android demo 中把 exact QR、四角 planar pose、target-front waypoint、center baseline、PnP controller 与
LOST/fresh-reacquire 接通，专项 JVM mechanics `8/8` 且 APK build 通过；但当前无 ready device，仍未运行真实相机，
也没有时延、光照/尺度/运动模糊或 18-run 指标。`NO_P1 / DEFAULT_APP_UNCHANGED` 继续禁止把旧 appearance lane
或这个未设备验证的 seam 晋升为导航、默认 App、安全或用户有效性。

Claim ceiling：

`CONTROLLED_DERIVED_DEVELOPMENT_DEMO_INDEPENDENT_VISIBLE_ANCHORS_NO_GENERAL_EXACT_INSTANCE_P1_NAVIGATION_SAFETY_OR_DEFAULT_APP_CLAIM`
