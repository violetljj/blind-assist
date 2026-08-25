# GRAIL M0 Oracle Interaction-Pose Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`REVERSIBLE_EXPLORATION / DEVELOPMENT_STANDARD / LAST_METER_ALGORITHM_MAINLINE_REOPENED / PROCEDURAL_METRIC_2_5D / ORACLE_REFERENT / ORACLE_GEOMETRY / UPPER_BOUND_ESTABLISHED / M1_AUTHORIZED / DEFAULT_APP_UNCHANGED`

## 问题

旧最后十米路线让目标框同时承担 referent、affordance、waypoint 和 arrival，单参考 exact-instance 与 V1-C/D/E/F 四边界路线都已按各自证据关闭。GRAIL 将任务改为：给定目标，输出一组可到达、目标可见、位于功能侧且朝向正确的 `站立位置 + yaw`，没有合法解时输出 `NONE`。

M0 不训练网络，只问这个新任务及其自动 teacher 在 oracle referent + oracle geometry 下是否存在稳定上界。因子明确分离为：

```text
referent != affordance != reachability != visibility != arrival
```

## 实现与 cohort

[`grail_m0.py`](../../../scripts/research/grail/grail_m0.py) 定义 metric 2.5D 建筑、目标功能侧、碰撞净空、视线、栅格可达性、set-valued pose teacher 与 `0.5 m / 20 deg` 位姿判定；[`run_grail_m0.py`](../../../scripts/research/grail/run_grail_m0.py) 生成并评估：

- 12 个 Development 建筑与 36 个 held-out 建筑，scene ID 和 target instance ID 全隔离；
- held-out 含 `door / counter / shelf / panel`，其中 24 个应有合法 pose set，12 个因功能侧被挡或隔墙而应为 `NONE`；
- 每个场景含一个同类错误实体；无人工逐帧标签；
- 四类反事实：同类错误实体、正确目标背面、自由但目标无关位置、面向目标但不可达位置；
- `B0` bbox + 固定距离、`B1` mask/depth + 最近自由点、`B3` oracle target + set-valued geometry。

这是 fresh 程序化 cohort，不复用 V1-F 的 ARKitScenes frame、boundary truth 或 outcome。它建立的是 task/teacher mechanics，不是自然 3D 或 RGB 泛化。

## held-out 结果

| 方法 | Interaction Pose Success | Closed-Loop Completion | No-Valid-Pose False Commit |
|---|---:|---:|---:|
| B0 bbox + fixed distance | 8/24 | 8/24 | 12/12 |
| B1 nearest reachable free point | 15/24 | 15/24 | 6/12 |
| B3 oracle set field | **24/24** | **24/24** | **0/12** |

Teacher 在 24 个 positive scene 的合法集合大小中位数为 23；对目标/几何 `4–5 cm` 微扰保持 set match 为 `24/24`。四类反事实均在全部 36 个 held-out scene 上实际构造并拒绝，分别为 `36/36`。所有预定 M0 门通过，终态：

```text
GRAIL_M0_PROCEDURAL_ORACLE_UPPER_BOUND_ESTABLISHED
```

B1 的 `15/24` 与 `6/12` false commit 是关键最小反例：可达自由点并不自动等于目标功能侧，也不能在当前无合法交互位姿时代表 `NONE`。

## 裁决、边界与 successor

最后十米算法主线以 GRAIL 新任务正式重开；旧 exact-instance matcher 和四边界/portal 表示保持关闭，不因 M0 恢复。V2-MARKER-POSE 只作隐藏的 `DEBUG / CALIBRATION / CONTROLLER CANARY`；动态风险是辅助能力。

唯一 successor 是 M1：冻结视觉编码器，在 building-disjoint 3D-derived Development 数据上比较 `B0 / B1 / B2 direct waypoint / GRAIL factorized set field`。M1 必须报告 interaction-pose success、wrong-target pose、no-valid-pose false commit 与 candidate permutation；没有清晰信号就不加长期记忆、主动搜索、Transformer zoo 或 Android。

Claim ceiling：程序化 metric 2.5D task、teacher、oracle planner 与闭环 mechanics。未建立 RGB、自然 3D scene、learned model、真实相机、Android、用户、产品或安全证据；默认 App 不变。

## 复现与本机证据

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_m0.py --output-dir artifacts.local/evidence/grail-m0
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover -s scripts/research/grail -p "test_*.py"
```

- `report.json`: `17B838E64D1F41AD2BFCA01B6078D108938BC8806EB64D3AF319149A6767DA1F`
- `rows.json`: `3DAC701EB39B627FB59EA32BD069BC1D8FEE8DD1FF714365BCC663C66AE287F7`
- `held_out_interaction_pose.svg`: `8089F644374E80D1CB376F7B6F4D3E613C357B6F1F78D94B1673A35066E78867`
