# TARO O0R factor-headroom runtime

状态：`current / TARO_RESEARCH_MODULE / PARTIAL_FACTOR_CANARY_COMPLETE / HISTORICAL_EVIDENCE_READ_ONLY / NO_ACTIVE_EXECUTION`

## 稳定 Interface

- `candidate_phase.py` 与 `depthart_runner.py`：truth-blind candidate materialization 和 completion seal。
- `factor_headroom.py`、`factor_canary.py` 与 `factor_evaluator.py`：固定 factor intervention、描述性 canary 与 evaluator。
- 本目录的 `test_*.py` 和 execution-preparation validator：验证冻结接口及两阶段证据防火墙。

## 输出

本地 candidate、factor rows、summary 与 receipts 只写入调用方显式绑定的 `artifacts.local/evidence/taro/` 子目录；仓库不接收模型输出、原始数据或 scientific evidence payload。

## 安全边界

该 Module 只保留已消费的 post-hoc WILD_LAB factor diagnostics。它不授权训练、重跑、选择阈值、产品晋升、设备部署或安全结论，缺失和不可评估状态必须保持 `UNKNOWN`。

## 停止条件

候选 seal、truth firewall、frame identity、factor ownership、预算或 receipt 任一不满足即停止。当前没有活动执行；动态权限只由 [`docs/research/taro/README.md`](../../../docs/research/taro/README.md) 维护。

This isolated runtime advances the admitted ARKitScenes R3 source/truth evidence
into a fixed DepthART-S factor-intervention experiment. It does not train or
modify DepthART.

The execution order is irreversible:

1. validate the signed R3 terminal, exact frame plan, source, checkpoint,
   runtime, implementation hashes, budgets, and absent factor evidence root;
   an R3 PASS authorizes formal factorial statistics, while the exact retained
   R3 `NOT_EVALUABLE` terminal authorizes only the descriptive canary below;
2. read only registered RGB and bound intrinsics for every exact eval frame;
3. run and seal every native 448x608 DepthART output plus its truth-blind input
   and inference receipt; preprocessing is the official RGB cubic lower-bound
   448/ImageNet path with row-scaled intrinsics. Output registration uses the
   same PyTorch bilinear `align_corners=True` operator, explicitly frozen on CPU
   so the persisted native raster deterministically reconstructs its high-res hash;
4. seal `candidate-phase-completion.json` before opening any per-frame truth;
5. re-fit the exact 211 ADAPTER_FIT frames and reproduce the R3 uncertainty
   receipt/artifact hashes (the compact artifact alone is audit evidence because
   its float cells are stored at the frozen 12-decimal canonical precision);
6. re-decode FARO/confidence for every available compact truth package,
   reconstruct all nine dense truth factor frames, and require every R3
   factor-frame and reducer-result commitment to match;
7. join the sealed candidate and always compute the threshold-free descriptive
   SCALE/SUPPORT/BOUNDARY canary. Only after an R3 PASS, and only for complete
   9/9 truth frames, run eight arms by two oracle modes by nine queries and
   compute the predeclared formal parent-macro statistics.

Within each query, the three parent frames are fully validated once and the
16 deterministic injections/reductions share that validated context. This is
hash/result-equivalent to the public per-arm path while avoiding repeated
million-point base-geometry validation that would exceed the wall-time lock.

`UNKNOWN` remains an explicit state. Geometrically known numeric truth may have
state `UNKNOWN` when its uncertainty interval crosses a decision boundary; it is
not discarded from the proper interval-score estimand. Candidate/extractor
failure produces explicit unknown arm rows and lowers coverage.

The experiment reuses the admitted R3 confidence/range uncertainty resolver for
both parents. It does not fit or claim a DepthART-specific calibration model;
the interval comparison is factor-headroom evidence, not absolute deployment
coverage evidence.

SCALE-containing arms use FARO geometry expressed in candidate gauge. This is
required because the frozen reducer applies SCALE to support height, observed
range, boundary XYZ, and their shape uncertainties; injecting absolute FARO
geometry together with SCALE would apply the metric correction twice.

The R3 retained `NOT_EVALUABLE` outcome therefore produces
`TARO_O0R_PARTIAL_FACTOR_CANARY_COMPLETE`, not a formal headroom PASS/FAIL. The
canary reports per-factor evaluability and parent-first descriptive medians; it
applies no threshold, hypothesis gate, or promotion rule. Missing FARO boundary
evidence remains `UNKNOWN` without erasing evaluable SCALE or SUPPORT evidence.

The claim ceiling is post-hoc WILD_LAB landscape-only factor diagnostics for
the exact locked ARKitScenes inputs and exact DepthART checkpoint. It
establishes no final task outcome, wearable, active-observation, device,
product, deployment, or safety result.

The one-shot budget is 8 hours wall time, 16 GiB host RSS, 8.5 GB maximum CUDA
allocation (the local RTX 5060 Laptop GPU reports 8151 MiB total), and 2 GiB of
scientific evidence. Network requests and training steps are both zero.

The source contract fixes every registered raster to landscape 1440x1920, so
the portrait-versus-landscape diagnostic is predeclared structurally not
applicable; no portrait sample is synthesized or rotated. The result claim is
therefore landscape-only. Other requested strata remain fail-closed when a
required comparison level or paired denominator is absent.
