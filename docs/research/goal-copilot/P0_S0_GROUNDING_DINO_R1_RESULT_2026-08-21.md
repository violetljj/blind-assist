# P0-S0-R1 Grounding DINO Tiny real-image materialization canary

状态：`COMPLETE / REAL_MAPILLARY_PROPOSALS_PASS / NOMINAL_SILVER_A_NOT_ACCEPTED / VISUAL_CROSSVIEW_CONSISTENCY_NOT_ESTABLISHED / NO_SCIENTIFIC_VERDICT`

## Outcome

`IDEA-Research/grounding-dino-tiny` 已在真实 Mapillary 图像上跑通，固定 revision、权重 hash、prompt、阈值、
逐图输入 hash、逐框 bbox/score/text label 与 runtime provenance 均已保存。模型只拥有
`VISUAL_PROPOSAL_ONLY`，训练数据 provenance 不完整只记录为 limitation，不再阻止 proposal generation。

首轮 30 张全 bbox 取样产生 217 个 proposals，但图像集中于无目标 anchor 的涂鸦巷，0 个进入 map/geometry
record，结果为 `P0_S0_PASS_WITH_COVERAGE_LIMITATION`。保留该结果后，只修正 source acquisition 为冻结的
anchor-facing 取图；模型、prompt 和阈值均未改。

anchor-facing run 使用 20 张真实图、7 个 Mapillary sequences，20/20 有 proposal，共 177 个。自动链得到：

- 122 个 proposal 射线命中没有唯一目标 crosswalk 的建筑；
- 20 个命中没有 admitted entrance anchor 的建筑；
- 17 个没有命中 60 m 内建筑；
- 14 个未过 3 m anchor / 2 m ambiguity gate；
- 4 个通过 map + geometry anchor，形成 1 个自动 multiview record；
- materializer nominal verdict：`P0_S0_MATERIALIZATION_CANARY_PASS`，`1 SILVER_A_PRIMARY`。

但结果后的可视一致性核对发现：唯一 primary record 的跨 sequence proposal 与另外 3 个 proposals 明显框在
不同的实体入口上；另外 3 张来自几乎相同的相机位置，不能脱离该冲突 view 单独满足 3 m baseline。现有
multiview mechanics 只比较 ray-wall 投影落点，没有验证 image-space region 是同一物理入口，因此这个 nominal
`SILVER_A_PRIMARY` 不接受进入 cohort。机器 verdict 与 receipt 保留，不事后覆盖；科学终态为：

> `VISUAL_CROSSVIEW_SAME_PHYSICAL_ENTRANCE_NOT_ESTABLISHED / NO_SCIENTIFIC_VERDICT`

这不是 Grounding DINO proposal failure：真实图 40/40 都产生 bbox。当前唯一已定位缺口是跨 view same-region
correspondence，不能再把“射线落到同一 map anchor”直接等同于“两个框是同一物理入口”。

## Frozen proposal configuration

- model: `IDEA-Research/grounding-dino-tiny`;
- revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`;
- `model.safetensors`: 689,359,096 bytes;
- SHA-256: `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`;
- license: model card `Apache-2.0`;
- prompt: `door . doorway . entrance . building entrance . storefront entrance . gate .`;
- box threshold: `0.15`;
- text threshold: `0.10`;
- class-agnostic deterministic NMS IoU: `0.50`;
- runtime: Python 3.11.9, PyTorch 2.11.0+cu128, Transformers 4.57.1, Pillow 12.2.0,
  NVIDIA GeForce RTX 5060 Laptop GPU.

Grounding DINO score 仅为 `MODEL_PROPOSAL_RANKING_SCORE_NOT_TRUTH`。它不能建立 entrance truth、target-building
truth、map/geometry/multiview truth、Silver quality 或 evaluator truth。

## Evidence

Task-owned ignored evidence root:
`artifacts.local/evidence/p0-s0/2026-08-21-grounding-dino-tiny-s0-r1-anchor-aware/`.

- proposal receipt SHA-256: `a70e9ec6e37f8d1371793d5cf940ea61ba6fe30b8c3baaf00aafec0b5caa351f2`;
- nominal materialization report SHA-256: `9efb129d51934410f84ef2f765fcea1d5e1c05b14dd9627ce191b75ad773e20d`;
- deterministic materializer replay: equal;
- visual audit: `post-run-visual-audit.json`;
- admitted-proposal contact sheet: `admitted-proposal-visualizations/contact-sheet.png`.
- visual audit SHA-256: `de6a22d5bb7fc7aa8f207539ea20f9ddff4f1b1cfbc121913cab4d3e04569798`;
- contact sheet SHA-256: `2416b0d7fb6b0e652f81ddeabfc042eb69c2ad01e635b09672d9664694f1f46e`.

## Next action and ceiling

不比较 detector、不换 Base、不调视觉阈值。最小下一步只修复现有 multiview mechanics：在授予
`MULTIVIEW_VERIFIED` 前，要求跨 view 的 image-space proposals 具有可审计的同一物理 region correspondence；
失败则保持 secondary/reject。修复后只在当前 consumed canary 做 Development replay，再决定是否建立新 cohort。

本轮证明真实 Mapillary image → Grounding DINO bbox/score/text proposal → map/geometry materialization 可运行，
并暴露了一个真实的 crossview identity 缺口。它不证明 Grounding DINO entrance recall/precision、合法 cohort、
grounding baseline、导航、安全、用户或默认 App 能力。
