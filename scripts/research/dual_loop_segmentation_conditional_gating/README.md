# dual_loop_segmentation_conditional_gating

状态：development；单一候选与阈值已冻结，尚未读取本协议结果

## 研究问题与版本

`DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0` 只回答：一个预先冻结的类别/机制条件门，
能否在不让任何 source-session recall retention 低于 `0.80`、不让
`boundary_step_curb` 或 `obstacle` 类 recall retention 低于 `0.80` 的情况下，同时达到
overall recall retention `>=0.90` 与 false-positive reduction `>=0.30`。

证据实例为 4 个 `r1_consumed_fresh`、4 个 dev、2 个 `consumed_old_blind` 的既有
520-frame consumed Development。十个 source-session ID 与 frame/image identity
互不重叠，但缺 participant、route 与 parent-capture ledger；因此这里只能做
source-session held-out stress reporting，不能称 LOSO independent validation。

## 稳定 Interface

```powershell
python -m scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating `
  --repo-root . `
  --config configs/dual_loop_segmentation_conditional_gating_r0/default.json `
  --output-root artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0/result

python -m scripts.research.dual_loop_segmentation_conditional_gating.validate_conditional_gating `
  --repo-root . `
  --config configs/dual_loop_segmentation_conditional_gating_r0/default.json `
  --evidence-root artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0/result `
  --output artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0/validation.json
```

输入 SHA、520 帧、11,757 个 raw component、十个 session 分母、单一候选与全部阈值
均由 config 绑定。runner 先复现 pilot/expansion 的 baseline、union causal 2-of-3 与
confidence `>=0.65` headline confusion，再生成任何条件门结果。

## 单一冻结候选

所有 temporal history 都按预测类别隔离，来自 raw class mask，并在 sequence/session
边界重置。`CAUSAL_2_OF_3` 的 pixel 支持是：

```text
T[k,t] = M[k,t] AND (M[k,t-1] OR M[k,t-2])
```

组件面积、置信度与上带 membership 只从 raw current component 计算一次；门后碎片不得
重算特征。上带 proxy 是纯几何 `INTERSECTS_UPPER_HEAD_BAND`（`y < 0.35H`，any pixel
intersection），不是 Atlas 中依赖 truth 的
`UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY`。

`CLASS_CONDITIONED_MULTI_NEGATIVE`

- obstacle 使用多重负证据 pixel rejection；
- boundary/step/curb 只整组件删除低置信小碎片，不因 temporal 或 upper proxy 单独删除。

其中 obstacle 的完整定义为：只删除 raw component 内同时满足
`noncausal pixel AND low confidence AND (small OR upper-intersecting)` 的像素。
文件提出的另两个候选只保留为未执行设计备选，不进入 config、runner、Pareto 或 terminal；
`每轮最多 3 个`是上限，而当前权威入口要求先冻结一个有限组合门。

缺失置信度不构成负证据，保留。候选 callable 不接收 truth、false activation、
mechanism tag、session、scene、role 或 YOLO attribution。

## 输出

仅写入指定 `artifacts.local/evidence/dual-loop-segmentation-conditional-gating-r0/`：

- `frame_metrics.jsonl`：baseline、两条 predecessor reference 与单一候选的逐帧
  overall/class confusion、post-component 与 false-component 计数；
- `component_decisions.jsonl`：每个 raw component 的单一候选 keep/reject/partial、
  causal-supported pixels、raw evidence、门后 fragment 数与拒绝原因；
- `result.json`：overall、逐 session、逐 role/scene、逐 class、source-session held-out
  stress、Pareto 与冻结 terminal；
- validator 输出：从逐帧/逐组件账本独立重算 aggregation、fold identity 和 terminal。

## 安全边界

本 Module 只读使用已查看/已消费 Development evidence。不训练，不访问 fresh holdout，
不接 Android、QNN、risk、feedback、TTS、振动或默认 App。visual-only sidecar 的
`drives_alerts=false` 保持不变。任何正结果最多支持未来另立 Confirmation 设计；
任何负结果只关闭当前 gating 路线并允许另立 residual-aware DDRNet Development。

## 停止条件

一次执行、一个候选、十个固定 source session 全量报告后停止。若候选没有同时通过五个
冻结门，则终态为
`CONDITIONAL_GATING_NO_ROBUST_INCREMENT_STOP_GATING_ROUTE`；不以未执行候选救援、
不改阈值、不按 fold 选候选。若候选通过，只记
`CONDITIONAL_GATING_ROBUST_INCREMENT_DEVELOPMENT_ONLY`，仍不产生提醒或产品权限。

## 假设与规则质疑

因 Atlas 的 background proxy 使用 truth，本协议明确拒绝把它直接接入 gate，替换为
运行时可得的纯几何上带 membership。LOSO 也被纠正为 fit-free source-session stress：
fold 只是逐 session 结果重排，必须满足 held-out result 与 direct session result
逐项一致，不能包装成独立泛化证据。

## 失败资产复用

失败结果可作为 residual-aware training task 设计、回归、counterexample 与
visual-only demo 依据；不得恢复十个 session 的 unseen/Confirmation 身份。
