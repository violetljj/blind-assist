# Semantic-Authority Conditioned Last-Mile Geometry V0

状态：`REVERSIBLE_EXPLORATION / V1_A_ALL_ORACLE_CEILING_PASS / V1_B_R2_B0_PASS / R3_CURRENT_CHAMPION / V1_C_PROXY_SUPERVISION_CLOSED / V1_D_ACTIVE_PARALLAX_CLOSED_RESCUE_0_OF_9 / V1_E0_ARKIT_MESH_TEACHER_LOW_CEILING_RESCUE_3_OF_9 / STOP_BEFORE_STUDENT / R6_NOT_RUN / B2_NOT_RUN`

本模块是 natural open-world SAGE-R 关闭后的唯一 Goal Copilot successor。身份由 QR 或 exact OCR semantic
authority 预先确认；geometry 只能回答目标 aperture 在哪里、如何接近以及是否具备到达证据，不能创建、替换或恢复 identity。

首轮比较固定为：

- baseline：semantic-anchor bbox center + bbox scale；
- SAGE-LM：0.24 m 主动视差、target aperture/approach point、LOST 时停步并等待 fresh semantic reacquire、连续两帧
  `near AND aligned AND aperture-supported` completion。

36 个 deterministic procedural episode 均衡覆盖 `ROOM_SIGN / QR_ENTRANCE / EXACT_SHELF_TARGET`。它只检验受控
几何机制，不是自然图像、真实设备、可通行性、安全、用户效果或产品 navigation 证据。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.experiment `
  --output-dir artifacts.local/evidence/sage-lm-v0/controlled-r1

E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.test_experiment
```

V1 已加入 `observation.py`、`rgb_observation.py`、`materialize_rgb_cohort.py`、`rgb_experiment.py`、
`render_rgb_demo.py` 与 truth-firewall 测试。24 个 curated ARKitScenes episode 的 exact anchor 为 controlled composited；
真实的是场景、边界与纹理。后续官方 pose audit 证明原 materializer 把 rotation-vector 列误作 camera positions，故原称
active pair 实测横向基线 `0.186–0.295 m` 无效。
结果 target-front arrival 为 baseline `7/24`、RGB SAGE-LM `2/24`，
未保留 V0 uplift；LOST 移动为 0。V1-A 随后在同 24 episode 用 evaluator truth 提供 aperture center/width/start range、
source camera positions 与 geometry confidence=1，原样进入同一 `_sage_lm()`；结果为 `24/24` target-front arrival、
`0.000 m` median lateral error、completion precision `24/24`、controls `6/6`，八条原标准全过。这只建立 frozen downstream
policy 的 all-oracle ceiling；下一步限于 source-pose-assisted two-view boundary geometry，不得改 policy、baseline、阈值、
cohort 难度或接 Android。

V1-B 已实现 B0/B1/B2 source-pose two-view line geometry，核心 path 不运行 LK 或 monocular metric depth；但冻结 active
pair 仅 `2/24` 满足原 motion gate，同 window 也只有 `13/24` 可找到任何合格替代帧，因此正式终态为
`NOT_EVALUABLE_SOURCE_POSE_PAIR_CONTRACT_INVALID`，B1/B2 raw output 不得当 boundary negative。future materializer 已改用
官方 camera-to-world inversion，但没有覆盖或重物化本 cohort。

V1-B-R2 随后从 source 新建 24 条正确 pose cohort，并在 outcome 前同时冻结 motion、第二视图 aperture projection
visibility 与最多 4 episode/source-sequence。24/24 pair gate 通过；B0=`24/24`，B1=`2/24` geometry、`0/24`
confidence/arrival，B2=`14/24` geometry、`5/24` confidence、`3/24` arrival。当前失败层为 RGB boundary candidate
extraction；B2 association 被上游 candidate recall 混杂，不作独立否定。

R3 使用官方 DeepLSD MegaDepth checkpoint 的 distance/orientation field，经过短 fragment fusion 后再形成 fitted boundary；
保留原 9 px localization、geometry、confidence 与 arrival。true pair=`15/24`、geometry=`13/24`、confident=`0/24`、
missing=`9/24`，四个门全未过。R4 在 x/depth grid 上联合两帧 field support 的 top-96 共享 3D boundary hypothesis，
退化为 true pair=`9/24`、geometry=`8/24`、confident=`1/24`、missing=`15/24`；joint-support objective rejected，B2 未运行。

R5 把 96 个 proposal 名额改成完整 aperture pair，并在四个投影 boundary x 上做 farthest-point diversity；结果 true
pair/geometry=`12/24`、missing=`12/24`，只部分救回 R4，仍低于 R3。512-pair diagnostic 可回到 `15/24`，说明
compression objective 仍是损失点，但不是正式成功臂。随后 11-fold leave-one-source-sequence-out 的小型 1D left/right
boundary head，held-out top-8 四边界覆盖仅 `5/24`，最终 true pair/geometry=`11/24`、missing=`13/24`，该
field-summary-only tiny head 被拒绝。dense support 与旧 confidence contract 不同，故 confident=`0/24` 不解释成全部
geometry 差；coverage 未过 18/24，R6/B2 均不运行。

V1-C 随后直接学习 RGB left/right boundary heatmap。TartanAir door-mask r1 在 synthetic validation 收敛但真实 cohort
为 C0/C1 true pair=`0/0`；排除 11 个评估 source 后，从余下 9 个 ARKitScenes sequence 的 1,350 帧自动生成 336 个
strong-line/depth-discontinuity opening proxy，按 7/2 sequence 训练/验证。同域 r2 的 C0/C1 四边界 Recall@8=`1/24`、
`4/24`，true pair/geometry=`1/24`、`3/24`，均显著低于 R3。当前 CNN + automatic opening-proxy supervision 被拒绝；
它不否定具有独立弱边界标签的大规模 task-specific supervision。R6/B2 继续不运行。

V1-D 保持相同 24 条、anchor、source pose、9 px 与 triangulation，只新增冻结 RAFT-Small 双向 flow、pose-derived
rotation compensation、forward/backward consistency 与 residual-parallax discontinuity。LEFT/RIGHT 各 top-8 的
四边界 Recall@8=`4/24`，true pair/geometry=`4/24`，R3 missing rescue=`0/9`；不做 R3 fusion，当前 parallax 实现关闭。
V1-E0 随后用官方 ARKitScenes 3DOD mesh + official pose/intrinsics raycast metric depth/normal，生成 RGB-independent
heatmap、signed depth jump 与 valid mask。四边界 Recall@8=`10/24`，true pair/geometry=`9/24`，R3 missing rescue=`3/9`、
retention=`6/15`；三个 teacher ceiling 门全部失败，故停止于 E0，不训练 student。该结果只覆盖全 24 条可用的 ARKit mesh
teacher；Faro-projected highres depth 因固定 pair 同时刻覆盖不足未形成完整 ceiling。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.materialize_rgb_cohort `
  --source-root artifacts.local/datasets/spatial-calibration-head-r1-arkitscenes-20260804/raw `
  --output-dir artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.rgb_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1/cohort.json `
  --output-dir artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.rgb_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1/cohort.json `
  --output-dir artifacts.local/evidence/sage-lm-v1a/all-oracle-ceiling-r1 `
  --observation-mode oracle

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.two_view_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1/cohort.json `
  --output-dir artifacts.local/evidence/sage-lm-v1b/source-pose-two-view-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.materialize_rgb_cohort `
  --source-root artifacts.local/datasets/spatial-calibration-head-r1-arkitscenes-20260804/raw `
  --output-dir artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.two_view_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --output-dir artifacts.local/evidence/sage-lm-v1b/source-pose-two-view-r2

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.dense_boundary_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --deeplsd-root artifacts.local/vendor/DeepLSD `
  --runtime-root artifacts.local/vendor/deeplsd-runtime `
  --checkpoint artifacts.local/vendor/DeepLSD/weights/deeplsd_md.tar `
  --arm b1 `
  --output-dir artifacts.local/evidence/sage-lm-v1b-r3/deeplsd-dense-boundary-b1-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.pose_accumulation_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --deeplsd-root artifacts.local/vendor/DeepLSD `
  --runtime-root artifacts.local/vendor/deeplsd-runtime `
  --checkpoint artifacts.local/vendor/DeepLSD/weights/deeplsd_md.tar `
  --output-dir artifacts.local/evidence/sage-lm-v1b-r4/pose-conditioned-accumulation-b1-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.anchor_pair_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --deeplsd-root artifacts.local/vendor/DeepLSD `
  --runtime-root artifacts.local/vendor/deeplsd-runtime `
  --checkpoint artifacts.local/vendor/DeepLSD/weights/deeplsd_md.tar `
  --output-dir artifacts.local/evidence/sage-lm-v1b-r5/anchor-conditioned-aperture-pair-b1-r3

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.boundary_head_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --deeplsd-root artifacts.local/vendor/DeepLSD `
  --runtime-root artifacts.local/vendor/deeplsd-runtime `
  --checkpoint artifacts.local/vendor/DeepLSD/weights/deeplsd_md.tar `
  --output-dir artifacts.local/evidence/sage-lm-v1b-r5s/sequence-disjoint-boundary-head-b1-r2

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.active_parallax_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --r3-report artifacts.local/evidence/sage-lm-v1b-r3/deeplsd-dense-boundary-b1-r1/report.json `
  --output-dir artifacts.local/evidence/sage-lm-v1d/active-parallax-boundary-field-b1-r2

uv run --python 3.11 --with open3d --with opencv-python-headless --with numpy --with pillow python -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.privileged_geometry_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1b/correct-pose-cohort-r2/cohort.json `
  --r3-report artifacts.local/evidence/sage-lm-v1b-r3/deeplsd-dense-boundary-b1-r1/report.json `
  --mesh-root artifacts.local/evidence/sage-lm-v1e/privileged-source/raw/Training `
  --output-dir artifacts.local/evidence/sage-lm-v1e/privileged-geometry-teacher-ceiling-e0-r1-r2-cohort
```
