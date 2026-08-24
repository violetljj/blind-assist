# Semantic-Authority Conditioned Last-Mile Geometry V0

状态：`REVERSIBLE_EXPLORATION / CONTROLLED_SYNTHETIC_GEOMETRY / ACTIVE`

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

当前唯一下一步：若 controlled uplift 明显成立，接入短 monocular RGB video 的真实 boundary/flow/depth observation，保留
完全相同的 identity firewall 与 baseline；不得把本 procedural result 写成真实场景 arrival。
