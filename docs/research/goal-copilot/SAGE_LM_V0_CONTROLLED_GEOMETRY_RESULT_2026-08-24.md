# SAGE-LM V0 Controlled Geometry Result

状态：`DEVELOPMENT_STANDARD / CONTROLLED_SYNTHETIC_GEOMETRY / AUTHORITY_SEPARATED_LAST_MILE / POSITIVE_MECHANISM_SIGNAL / REAL_RGB_NOT_RUN / DEFAULT_APP_UNCHANGED`

## 结论

natural open-world SAGE-R 保持 `CLOSE_NATURAL_SAGE_R`，不再 rescue。唯一 successor 改为
`SEMANTIC_AUTHORITY_CONDITIONED_LAST_MILE_GEOMETRY_V0`：QR/exact OCR 只回答 **WHO**，几何只回答 aperture、approach point、
progress 与 completion evidence，几何不得创建或替换 identity。

首轮 deterministic controlled experiment 使用 36 个 procedural episode，均衡覆盖 `ROOM_SIGN / QR_ENTRANCE /
EXACT_SHELF_TARGET`。同一已确认 semantic anchor 下，baseline 追 anchor bbox center 并以 bbox scale 到达；SAGE-LM 由
0.24 m 双视点的 noisy aperture-boundary bearings 三角化 approach point，semantic LOST 时不移动，fresh reacquire 后继续，
并要求连续两帧 `near AND aligned AND aperture-supported` 才确认到达。

| 36 episodes | bbox center + scale | SAGE-LM V0 |
|---|---:|---:|
| correct direction | 22.9% | **67.9%** |
| target-front arrival | 7/36 (19.4%) | **33/36 (91.7%)** |
| median endpoint lateral error | 0.592 m | **0.094 m** |
| verified completion | 7/36 | **28/36** |
| completion precision | 19.4% | **93.3%** |
| premature arrival | 29 | **2** |
| LOST recovery | 12/12 | 12/12 |
| movement steps while semantic LOST | 24 | **0** |

这是明显的受控机制正信号：semantic carrier 和 physical approach region 不重合时，追门牌/QR/标签中心会系统性把终点拉偏；
显式 target-aperture geometry 能恢复大部分方向、终点和 completion precision。剩余 3 个 endpoint failure 与 6 个未确认 episode
说明 noisy short-baseline triangulation 仍有观测/置信度问题，当前结果不是被做成完美成功。

## 证据边界

本轮只使用 procedural geometry、synthetic noisy bearings、已知 0.24 m camera translation 与 simulator arrival truth。没有真实 RGB、
光流/深度网络、真实相机运动估计、门开闭/连通性、障碍物、人体实验、设备闭环或安全证据。它证明的是 authority separation 加
active aperture/progress 表示在受控任务定义中相对 bbox baseline 有信息，不证明真实场景 last-mile navigation 或“可通行”。

它也不恢复旧 synthetic current-frame arrival 的产品证明责任：旧 S2-S5/D1C/visual-servo 终态保持不变。本轮改变了输入合同和研究问题，
且 claim ceiling 为 `CONTROLLED_GEOMETRY_MECHANISM_EFFECT_ONLY`。

## 复现

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.experiment `
  --output-dir artifacts.local/evidence/sage-lm-v0/controlled-r3

E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.test_experiment
```

- `report.json` SHA-256: `8fd3186670dd3e34a37183fe6eed75ef46db5bc37d8e33a314d3f55efdaf2de4`
- `trajectory_demo.png` SHA-256: `98de761444665a95a6411363a892d7fcadcace03fedd834b0923640cd4e69ba4`

## 唯一下一步

保持同一 identity firewall、baseline 和指标，接入 controlled short monocular RGB video 的真实 boundary/flow/depth observation adapter；
先检验 observation 能否复现 aperture/approach uplift，再决定是否加入 learned encoder 或 Progress Belief。当前不做 UI、Android、VLM、
generic identity、动态风险、完整 navigation 或默认 App 集成。
