# SAGE-R V2.1 RapidOCR Transfer Result

状态：`DEVELOPMENT_STANDARD / GENERATED_PIXEL_SCENE_RAPIDOCR / FROZEN_V2 / CORRECT_1_TO_9 / WRONG_LOCK_4_TO_0 / NONE_0_TO_3 / UNKNOWN_0_OF_2_TO_2_OF_2 / NATURAL_PHOTO_NOT_EVALUATED / DEFAULT_APP_UNCHANGED`

## 结论

固定的 SAGE-R V2 scorer、belief 更新和全部阈值已原样接到 RapidOCR 3.9.2 的真实检测/识别输出。16 张自动生成的门牌式像素图覆盖
`301 / 302 / 320 / 302A`、directory board、partial OCR、target absent、blur 和 correlated burst；RapidOCR 实际输出 112 条
识别 line 及 quadrilateral polygon、confidence，适配器再把这些像素坐标归一化为 V2 的 token/candidate geometry。

| 指标（16 generated-pixel RapidOCR frames） | substring + 2-frame FSM | frozen SAGE-R V2 | delta |
|---|---:|---:|---:|
| correct terminal frames | 1 | **9** | **+8** |
| target correct locks（9 target-present） | 1 | **6** | **+5** |
| wrong locks | 4 | **0** | **-4** |
| correct `NONE`（7 absent） | 0 | **3** | **+3** |
| low-observability `UNKNOWN` preserved | 0/2 | **2/2** | **+2** |

这说明 V2 的 relational signal 不只存在于手填 token 表：在 RapidOCR 的 detection、line segmentation、confidence 和 polygon
抖动进入后，同一受控 cohort 上仍明显优于 substring/FSM。最强反例仍是 directory：RapidOCR 在 directory 上确实读出
`ROOM 302`，baseline 在第二帧把它锁到最近的 A，并在四帧重复 burst 上继续错锁；V2 首帧保持 `UNCERTAIN`，下一新视角由
B 上的 partial `30` 关系证据锁定 B，而 absence burst 始终不锁任何门，fresh wide view 转为 `NONE`。

## 适配边界

- RapidOCR 原始 text、confidence、quadrilateral polygon 和图片 SHA 全量保留在 `raw-decisions.json`。
- RapidOCR 把同一行输出成带空格的字符串时，适配器按字符长度切分 line polygon；不改识别字符。
- 对唯一匹配目标 token 的长度至少为 2 的严格前缀，适配器把 raw `30` 表示成 V2 已有语义的 `30?`；raw `30` 仍保留。这是固定、可审计的 OCR-to-V2 interface 语义，不是改 scorer 或扫 threshold。
- candidate geometry 来自生成器的门框，经与图像相同的透视变换后取 axis-aligned box；它不是 detector measurement。
- blur 由输出图像 Laplacian variance 映射；perspective 来自生成参数。两者不是自然相机 calibration。

V2 source SHA-256：`c6ecb7625c996982b04134c2824c8efe6026dab7da49b4c13c3a4893a28b95cb`。运行报告将 source path/hash 写入结果，避免把 adapter
误写成算法修改。

## 失败归因

本次 16 帧中，V2 的 target/NONE/UNKNOWN/UNCERTAIN 行均满足预期或得到更强的正确 target terminal，因此没有产生可归因的
错误帧。runner 已实现只对真实不满足行给出 `OCR_TOKEN_GROUPING / CANDIDATE_ASSOCIATION / OBSERVABILITY /
LEXICAL_CORRUPTION / BELIEF_ACCUMULATION` 归因；本轮 `failure_classes={}`，不能据此声称这些 failure mode 已在自然照片上解决。

## 复现

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_semantic_anchor_graph_and_belief_v2_1_real_ocr `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.test_semantic_anchor_graph_and_belief_v2

E:\codex-tools\bin\blindassist-python.cmd -m `
  scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1.semantic_anchor_graph_and_belief_v2_1_real_ocr `
  --run-dir artifacts.local/evidence/semantic-anchor-graph-belief-v2-1-real-ocr/run-20260824T223500+0800
```

Focused tests：`9/9 PASS`。Evidence：
`artifacts.local/evidence/semantic-anchor-graph-belief-v2-1-real-ocr/run-20260824T223500+0800/`。

| 文件 | SHA-256 |
|---|---|
| `raw-decisions.json` | `2396161000624b604936aa58e6baf8ee6ef63bae26af1ed9860210df67da081a` |
| `final-report.json` | `52fa2834788eefdb701f6f4ed023a81d0efb70a2f1da847ac0dc2fb50420f08a` |
| `result.html` | `f8765857fb0d96c0d3570a59d97a14c29fc1a3412d6f9691f63559f554b008ba` |

## 下一动作与 claim ceiling

V2.1 已满足“固定 V2 接真实 OCR polygon”的受控 transfer 问题。下一算法动作可进入小型 learned relational scorer（V3）：
保留显式 semantic graph 与 open-set terminal，用 domain-randomized graph training 学习 relation score，并把自然照片门牌/directory
作为未参与训练的 Development test。V4 active information gain 仍不启动。

Claim ceiling：

`GENERATED_PIXEL_SCENE_RAPIDOCR_DEVELOPMENT_TRANSFER_NO_NATURAL_PHOTO_CAMERA_OPEN_WORLD_CALIBRATION_ANDROID_NAVIGATION_SAFETY_OR_PRODUCT_CLAIM`
