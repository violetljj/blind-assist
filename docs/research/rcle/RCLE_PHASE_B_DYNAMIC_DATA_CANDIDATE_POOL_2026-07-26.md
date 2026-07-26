# RCLE Phase B 动态数据候选池

状态：`DISCOVERY_INPUT / NOT_ADMITTED / DYNAMIC_PRIORITY`

日期：2026-07-26

来源输入：`D:\edge\可能对算法研究有帮助的数据清单.md`

适用协议：
[Phase B 渐进式协议](RCLE_PHASE_B_PROGRESSIVE_PROTOCOL_2026-07-26.md)

## 结论

外部 GPT 清单提供了有价值的候选，但它是待验证假设池，不是数据准入结果、固定下载
队列或 confirmation split。排序可根据 PB-H1 结果、本地已有缓存、下载成本、来源
可达性和新证据动态调整。

当前最值得保留的候选是 TUM `fr2/rpy`、ETH3D `sofa_shake`、ICL-NUIM
`lr kt2/kt3` 和 ETH3D `sofa_3/4`。其中只有来源能力与自然语言场景描述得到局部
核验；“纯旋转”“强接近”及具体窗口仍必须由 pose + depth 连续几何复算，不得按
sequence 名称直接授予角色。

## 当前证据分级

| 候选 | 当前角色 | 已知价值 | 仍需验证 | 当前处置 |
| --- | --- | --- | --- | --- |
| TUM `fr2/rpy` | Rotation discovery 首选 | 官方 RGB、注册深度、mocap pose；整体低平移、缓慢 RPY 旋转 | 逐窗口 translation-induced expansion、parallax、depth coverage | `P1` |
| ETH3D `sofa_shake` | Rotation discovery 第二候选 | 官方 RGB-D、同步图像、ground-truth pose；描述为在沙发前旋转 | 逐窗口平移污染、静态表面 closing rate、有效时长 | `P2` |
| ICL `lr kt2` / `lr kt3` | Synthetic approach calibration/canary | 精确合成 depth、pose 和场景几何 | 是否存在足够持续的光轴接近；两轨迹的增量独立性 | `P3_ONE_ONLY_FIRST` |
| ETH3D `sofa_3` / `sofa_4` | Real-domain approach discovery | 真实同步 RGB-D 与 pose；独立 sequence | “在沙发前移动”是否实际形成强接近，不得由名称推断 | `P4_ONE_ONLY_FIRST` |
| TUM `fr2/xyz` | Approach 备选 | 注册 RGB-D、mocap pose、明显平移 | 平移是否沿光轴、主表面 closing rate 是否足够 | `RESERVE` |
| ETH3D `camera_shake_2` | Rotation stress/counterexample | 快速 shake、RGB-D、pose、IMU | 平移和运动模糊可能同时存在，不适合优先充当干净旋转 | `STRESS_ONLY` |
| EVIMO2 | 后期困难域压力测试 | RGB、Vicon pose、逐像素 depth、多种运动 | 数据规模大、truth 不规则、动态物体和低照度混淆 | `DEFER_TARGETED_SEQUENCE_ONLY` |
| Bonn frozen cohort | Regression/source characterization | 已有本地资产、同步和失败路径 | 已发生 outcome access，不能恢复为 unseen confirmation | `REUSE_ONLY` |

`P1–P4` 只是当前信息增益顺序，不是硬门。若本地已有某候选的完整、hash-bound
payload，或小型 pose 审计快速否定上一候选，可以调整顺序并记录原因。

## 已核验的来源事实

- TUM 官方提供注册 RGB/depth、mocap ground truth；`fr2/rpy` 官方整体统计约为
  `0.014 m/s` 平移和 `5.774 deg/s` 旋转，但 aggregate 不能代替逐窗口角色审计。
- ETH3D 官方提供同步同视点 RGB-D、ground-truth pose；`sofa_shake` 的官方描述是
  相机在沙发前旋转，`sofa_3/4` 只描述为在沙发前移动。
- ICL-NUIM living-room `kt0–3` 提供合成 depth、camera pose 和场景几何；
  `kt2/kt3` 的“强 approach”仍是待证假设。
- EVIMO2 完整 v2 解压规模约 `525 GB`，传统 RGB 相机部分约 `271 GB`，且 Vicon
  遮挡会使 depth/mask truth 非均匀采样，因此当前禁止整库下载式探索。

官方入口：

- TUM RGB-D：
  <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download>
- TUM 文件格式：
  <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats>
- ETH3D SLAM 数据：
  <https://www.eth3d.net/slam_datasets>
- ETH3D 格式：
  <https://www.eth3d.net/slam_documentation>
- ICL-NUIM：
  <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>
- EVIMO2：
  <https://better-flow.github.io/evimo/docs/ground-truth-format.html>

## 高效使用规则

1. 在任何新下载前先运行 PB-H1；当前几何代理未厘清时，增加数据不会区分代理错误与
   来源缺失。
2. 每次只推进一个候选：先检查本地缓存和来源 metadata，再做小型 pose-only 审计，
   只有通过后才取得所需 depth/RGB payload。
3. ICL `kt2/kt3` 和 ETH3D `sofa_3/4` 各先选一条；第一条不能提供角色或新信息时，
   不机械下载同组第二条。
4. Discovery 保存连续分布与缺失原因，不把继承的 `0.02 m/s` 或 `0.05/s` 自动写成
   新硬门。
5. 看过 geometry 或算法 outcome 的 sequence 必须记录 access level；角色变化不能
   把它重新包装成独立 confirmation。
6. EVIMO2 只允许 sequence-level 定向选择；在前面低成本候选未解决问题前，不下载
   完整集合。

## 当前后继

唯一立即执行项仍为 `PB-H1-ROLE-PROXY`：

1. 建立一个纯旋转、无真实接近的受控合成 fixture；
2. 实现 raw translation speed、pose+depth translation-induced radial expansion
   与 time-normalized parallax；
3. 先做物理标定，再在一个已烧掉的 Bonn diagnostic window 上比较；
4. 输出 proxy mismatch、真实平移污染或实现错误三者的区分结论。

只有 PB-H1 形成 round summary 后，才按本候选池选择第一个新来源。默认首选 TUM
`fr2/rpy`，但若本地缓存与取得成本改变，允许以记录理由的方式调整。
