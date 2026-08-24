# Semantic-Authority Conditioned Last-Mile Geometry V0

状态：`REVERSIBLE_EXPLORATION / V1_RGB_ADAPTER_FAIL / V1_A_ALL_ORACLE_CEILING_PASS / V1_B_R2_B0_PASS / V1_B_R2_RGB_BOUNDARY_EXTRACTION_FAIL / ASSOCIATION_UNADJUDICATED`

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
```
