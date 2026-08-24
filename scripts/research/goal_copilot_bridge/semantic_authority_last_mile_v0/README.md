# Semantic-Authority Conditioned Last-Mile Geometry V0

状态：`REVERSIBLE_EXPLORATION / CONTROLLED_REAL_RGB_OBSERVATION_V1_FAIL / OBSERVATION_DIAGNOSTIC_ONLY`

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
真实的是场景、边界、纹理、运动与深度现象。轨迹筛选保证 active pair 的实测横向基线为 `0.186–0.295 m`。
结果 target-front arrival 为 baseline `7/24`、RGB SAGE-LM `2/24`，
未保留 V0 uplift；LOST 移动为 0。当前只允许分解 boundary association、reciprocal flow survival 与 metric-depth range，
不得改 SAGE-LM policy、baseline、阈值、cohort 难度或接 Android。

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.materialize_rgb_cohort `
  --source-root artifacts.local/datasets/spatial-calibration-head-r1-arkitscenes-20260804/raw `
  --output-dir artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.semantic_authority_last_mile_v0.rgb_experiment `
  --cohort artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1/cohort.json `
  --output-dir artifacts.local/evidence/sage-lm-v1/controlled-real-rgb-r1
```
