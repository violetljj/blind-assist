# Failure-Aware Causal Component Validator R0 Result

状态：`COMPLETE / VALID / NOT_SUPPORTED_AND_GATING_STOP /
CONSUMED_DEVELOPMENT_CROSSFIT_ONLY / ACTIVE_SEGMENTATION_GATING_STOP /
FINAL_CONFIRMATION_NOT_ACTIVATED / DEFAULT_APP_UNCHANGED`

日期：2026-08-01（Asia/Hong_Kong）

协议：`DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0`

## 结论

冻结的轻量 learned component validator 不受支持。10 个 source-session 的 nested
LOSO cross-fit 只通过 4/9 项 utility 门，且 host P95 增量 latency 失败：

- FP reduction `0.177920 < 0.30`；
- overall recall retention `0.855661 < 0.90`；
- minimum-session recall retention `0.466375 < 0.80`；
- boundary retention `0.207740 < 0.80`；
- `C-A` FP-area increment `0.087407 > 0.05`；
- host P95 incremental latency `9.376145 ms >= 3 ms`。

因此冻结终态为：

```text
NOT_SUPPORTED_AND_GATING_STOP
```

这不是 near miss。候选失败 5 项 utility 门，工程 latency 也失败；retained false area
中 `STABLE_HIGH_CONFIDENCE_ERROR` 只占 `0.373382 < 0.50`。本 R0 不授权
Component-aware Loss DDRNet successor，不换 XGBoost/MLP/Transformer、不改 feature
subset 或阈值救援。当前 reference 上的 active learned segmentation gating 关闭；
分割只保留 visual sidecar / coverage diagnostic。

## 冻结身份与执行

- 冻结 Git commit：
  `ca503f3ee854a701e932f2d12e81559b9a2d122f`
- config SHA-256：
  `31b7f5267ee2a9fdffc4eb42762a29aecb14be373f3285ba258c3f5c5ba8d9c3`
- 输入：520 帧、11,757 raw components、10 个 source-session
- 标签：4,230 `KEEP` / 7,527 `REJECT`
- feature：21 列 current/past runtime allowlist
- 模型：唯一 `StandardScaler + L2 Logistic Regression`
- outer：10-fold leave-one-source-session-out
- inner：每个 outer training context 内 9-fold LOSO
- threshold：固定 `0.05..0.95`、步长 `.05`
- fresh holdout：未访问

每个 outer-heldout session 均从自身 fold 的 scaler、sample weight、模型与 inner
threshold 中排除。10 个 fold 选出的 threshold 为 `.70..0.85`；相应 inner OOF operating
point 也只通过 4 或 5 项门，最小规范化 margin 为 `-.867845..-.608119`，不存在训练上下文内
的全门 operating point。

全部 session 的 outcome 过去都已被研究流程查看，且缺 participant、route、
parent-capture identifiers。因此本结果只叫
`CONSUMED_DEVELOPMENT_CROSSFIT_ROBUSTNESS`，不叫 fresh、unseen、independent
validation 或 Confirmation。历史 R1 amendment 与 terminal 均未修改。

## 九门结果

| gate | value | threshold | result |
|---|---:|---:|:---:|
| FP reduction | .177920 | >= .30 | FAIL |
| overall recall retention | .855661 | >= .90 | FAIL |
| minimum-session recall retention | .466375 | >= .80 | FAIL |
| boundary retention | .207740 | >= .80 | FAIL |
| obstacle retention | .908639 | >= .80 | PASS |
| C-A recall | .262583 | >= .05 | PASS |
| C-A FP-area increment | .087407 | <= .05 | FAIL |
| component recall | .543340 | >= .50 | PASS |
| false components/frame | .478846 | <= 3.0 | PASS |

最弱 session 为 `lmkIchCJ1RIKsZvbb4HjCDl85B2nOicv`，recall retention
`0.466375`；其余 session 范围为 `0.712916..0.950170`。这不是只差整体阈值，而是明显
的跨 session 与类别失衡。尤其 boundary retention 只有 `0.207740`，同时 obstacle
retention 为 `0.908639`；组件分类器没有形成可泛化的 hazard-component quality 判断。

## 冻结对照

| arm | FP reduction | overall | min-session | boundary | obstacle | C-A recall | C-A FP area | comp recall | false comp/frame | passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| causal 2-of-3 | .297925 | .776650 | .472921 | .368111 | .810154 | .238337 | .074648 | .612390 | 8.871154 | 3/9 |
| confidence >= .65 | .304290 | .795422 | .379836 | .311604 | .839183 | .244097 | .073971 | .465720 | .509615 | 4/9 |
| historical primary conditional | .092572 | .942399 | .774580 | .945451 | .946764 | .289201 | .096482 | .674094 | 6.300000 | 5/9 |
| learned validator | .177920 | .855661 | .466375 | .207740 | .908639 | .262583 | .087407 | .543340 | .478846 | 4/9 |

learned validator 把 false-component count 压得很低，但通过大量拒绝 boundary truth
换取该结果。它相对 historical primary 获得更多 FP reduction 和更少 false components，
却显著损失 overall、minimum-session 与 boundary retention；相对 confidence gate 又
没有达到 `.30` FP reduction。它不支配任何 reference，也没有产生新的受约束 Pareto
operating point。

## 系数与消融诊断

10 个 outer fold 中，平均绝对 standardized coefficient 最大的前五项为：

| feature | mean coefficient | mean absolute coefficient | sign consistency |
|---|---:|---:|---:|
| bbox width fraction | .553017 | .553017 | 10/10 positive |
| obstacle × near YOLO union | .496277 | .496277 | 10/10 positive |
| top-1 confidence median | .484233 | .484233 | 10/10 positive |
| near YOLO union <= 3 px | -.440324 | .440324 | 10/10 negative |
| top-1/top-2 margin median | -.357372 | .357372 | 10/10 negative |

这些只是 cross-fit 线性关联，不是因果机制证明。固定 zero-block、无 refit、无
rethreshold 消融中，删除 geometry block 把 FP reduction 提高到 `.728`，但 overall /
minimum-session / boundary retention 同时降到 `.373 / 0 / 0`；其他 block deletion
仍只通过 4/9 门。没有某个单 block 能解释或救援全门失败。

保留的 333 个 false components、403,857 pixels 中，Atlas
`STABLE_HIGH_CONFIDENCE_ERROR` 为 66 components、150,793 pixels，面积占比
`.373382`，覆盖 7 个 session。该标签只作 outcome 后诊断，没有进入 feature matrix；
占比未达到 near-miss 预冻结的 `.50`。

## 工程门

benchmark 在 Windows host / Python 3.11.9 / NumPy 2.1.3 上预缓存 raw components 与
masks；timed region 只含因果特征、标准化、sigmoid、keep/reject 与 class-label mask
重建，不含 DDRNet/YOLO inference、truth、文件 I/O 或 fit。

| engineering check | value | threshold | result |
|---|---:|---:|:---:|
| host P95 incremental latency | 9.376145 ms | < 3 ms | FAIL |
| serialized model + scaler | 1,847 B | <= 65,536 B | PASS |
| bounded state + feature buffer | 1,000,023 B | <= 1,048,576 B | PASS |

P99 为 `11.985822 ms`。内存轴满足冻结上限，但当前 host reference 的增量延迟不满足
utility 通过后才能进入平台 benchmark 的前置门。不得启动 Snapdragon 8 Gen 2 / A568
算法比较；该 host 结果也不构成 Android/QNN 性能结论。

## 验证与绑定

validator 从 bound inputs 重建全部 component table，检查 exact feature allowlist 与
truth/future firewall，复核 10 个 outer / 90 个 inner split，使用纯 NumPy 重新计算
11,757 个 held-out probabilities，再从 component decisions 重建 520 帧账本、九门、
P95、内存和 terminal。

- validator：`VALID`
- top-level checks：9/9
- historical comparator exact integer checks：27,040
- runtime feature/prediction checks：11,757
- component table SHA-256：
  `b38ee74b43f68262e1f1f40886c94a544aad4a1baf275d3f2cc3f9e7cb2da960`
- evaluation result SHA-256：
  `f41ef2721622f6c3e5d5e2ac028917c97caf7a67dd620c2592d232fecd3fb271`
- fold models SHA-256：
  `0caef35b154be299036dcd21089b9a7e3d1fc2fddccc1c1605e930a122f174f8`
- held-out predictions SHA-256：
  `9bed742a5c788ea9000e5b5ee80c4a781e7486d9f5ca1524565b2de5432f9b80`
- frame metrics SHA-256：
  `f6e0d0fc719a0d8f83ac738acea994d525a488cd162a237dc0d04a5163b13821`
- benchmark report SHA-256：
  `113f31f6b5ff85ddffeb4ae5fcb3afbd77a58f738094d87c19aa5aa0339a14bb`
- validation SHA-256：
  `fc5ad33e608a316d74b16bf7af4d5dd1f7cb727d3ce640ff9eda772ddc05fd5c`

## 停止与资产复用

本 R0 之后禁止：

- 在同一 520 帧上改 threshold grid、C、solver、seed 或 feature subset；
- 换 XGBoost、MLP、Transformer gate 或选择少数 session/fold；
- 把 diagnostic mechanism tags 反灌为 inference feature；
- 启动 Component-aware Loss DDRNet、INT8/QNN、A568、Android active fusion；
- 接 risk/feedback、TTS、振动、提醒或默认 App。

负结果保留为 learned-gating counterexample、component-aware training 的未来设计约束
和 visual/coverage diagnostic。若未来重开分割训练，必须由新的数据与协议提出不同
因果变量；本结果不提供自动 successor authority。
