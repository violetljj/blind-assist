# Active Distinctive Evidence Acquisition V0 Result

状态：`CONTROLLED_DEVELOPMENT / CURATED_4_TARGET_16_DECISION_DEMO / APPEARANCE_DERIVED_DISTINCTIVE_ANCHOR_NO_UPLIFT / SEMANTIC_ANCHOR_NOT_EVALUABLE_NO_OCR_RUNTIME / PASSIVE_MAINLINE_REMAINS_STOPPED / NO_DEFAULT_APP_CHANGE`

## 结论

V0 已把输入从单张 reference 改为三帧 reference sweep，并在两个真实公开 storefront、一个包装商品和一个个人物品上
完成受控 demo。最终 4 targets × 4 target-present views 共 16 次 paired decisions，另插入 4 次 target-lost / fresh
reacquisition 机会。结果没有产生可展示增益：

| 指标 | passive single-reference DINO | active distinctive-anchor | delta |
|---|---:|---:|---:|
| target top-1 | 11/16 | 11/16 | 0 |
| wrong-target lock | 9/20 | 9/20 | 0 |
| reacquisition | 3/4 | 3/4 | 0 |
| median time-to-first-lock | 0.0 s | 0.0 s | 0.0 s |

三个场景也逐项相同：store/entrance=`4/8 top-1, 6 wrong locks, 1/2 reacquisition`，product=`3/4, 2, 1/1`，
personal item=`4/4, 1, 1/1`。因此当前 appearance-derived patch anchors 不是 successor：它只是把同一个 DINO
appearance signal 改写成稳定 patch 与 candidate-unique voting，没有获得新的 identity information。

## 本次与 passive mainline 的实质差异

- storefront 用公开 reference 的确定性小幅旋转/缩放模拟三帧短扫；Washington 商品/个人物品用同一真实视频早段的
  `q=.05/.15/.25` 三帧 enrollment sweep，搜索只用后段 `q=.35/.50/.65/.80`；
- SIFT 只检查 reference sweep 是否存在跨视角稳定局部点；DINO patch anchor 必须在至少两个 reference views 重复；
- 每个 anchor 只能投给绝对 similarity 足够且相对其它候选有 margin 的一个 candidate；没有
  `max(reference1, reference2, reference3)`；
- active arm 必须同时满足 distinctive-anchor 数和 candidate score margin 才 lock；tracker 没有 identity authority，
  本轮也没有实现 tracker。

这确实改变了输入合同，但没有改变有效表示；所以它不推翻此前 passive stop，也不能靠继续调 cosine、margin、SIFT、
crop 或 lock threshold rescue。

## 执行过程

第一次运行在指标前拒绝 `cell_phone`：三帧真实 sweep 只有 5 个稳定 SIFT anchors，低于固定的 12。替换为同属
personal-item demo 且通过同一 acquisition front door 的 `keyboard` 后，R1 完成。R1 的 SIFT-only gate 把 wrong lock
从 11 降至 0，但同时产生 0 次 lock、0/4 reacquisition，属于纯弃权，不是 uplift。

唯一一次机制修订 R2 改为：SIFT 只审计 acquisition，稳定 DINO local patches 承担 candidate-unique anchor voting；同时
Washington reference sweep 改为同一 capture 的早段，明确放弃 independent-session exact-instance claim。R2 产生上述
最终指标。观察到与 passive 三个总指标、三个场景分层全部完全相同后停止，不再调门。

运行命令：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.active_distinctive_evidence_acquisition_v0 `
  --repo-root . `
  --model-dir artifacts.local/models/p1_a2_dinov2_small_ed25f3a `
  --run-dir artifacts.local/evidence/active-distinctive-evidence-acquisition-v0/run-20260824T100000Z-r2 `
  --device cuda
```

证据目录：`artifacts.local/evidence/active-distinctive-evidence-acquisition-v0/run-20260824T100000Z-r2/`。

| 文件 | SHA-256 |
|---|---|
| `cohort-manifest.json` | `56093a8f641537b9ebdd0cd67fd26d4be77f4e17aca0d4498821bc50746ad588` |
| `raw-decisions.json` | `8379bffc19ee157747ccfc867bb95fec7bb0516b7fb543563961f5352431cad0` |
| `final-report.json` | `555651c419e22fb7ce2514ea8e95e7ef754723afe2a0ea3ba7377d70221257c8` |

## 作用域解耦与下一动作

`NO_P1 / DEFAULT_APP_UNCHANGED` 只禁止把失败的 passive exact-instance verifier 或本次无 uplift arm 晋升为 identity
authority；它不禁止 BlindAssist 继续做受控展示原型。tracker 仍可在未来由独立 anchor 确认后承担短时连续性，但不能
自己建立或恢复 identity。

本机当前没有可调用的 PaddleOCR、EasyOCR、Tesseract 或 RapidOCR runtime；既有 PP-OCRv5 输出只覆盖旧 canary 的部分
storefront frames，不能补成当前 16-decision cohort，故 OCR/logo semantic arm 为
`NOT_EVALUABLE_NO_EXECUTABLE_OCR_RUNTIME`，不是负证据。唯一仍有信息增益的 successor 是可执行后再另立的
`SEMANTIC_DISTINCTIVE_ANCHOR_V1`：要求 location-/package-/owner-specific OCR、logo 或 marker 提供独立信息；禁止再调
appearance patch matcher、aggregation 或 lock threshold。

Claim ceiling：

`CONTROLLED_DEVELOPMENT_DEMO_WITH_CURATED_VISIBLE_DISTINCTIVE_ANCHORS_NO_GENERAL_EXACT_INSTANCE_P1_NAVIGATION_SAFETY_OR_DEFAULT_APP_CLAIM`
