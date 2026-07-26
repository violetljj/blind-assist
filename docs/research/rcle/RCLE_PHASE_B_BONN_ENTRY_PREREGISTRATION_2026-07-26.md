# RCLE Phase B Bonn 入口预注册

状态：`DESIGN_FROZEN / DESIGN_REVIEW_PASS / EXECUTION_NOT_AUTHORIZED`

日期：2026-07-26

## 结论

Observable Support Recovery R0 的独立 synthetic validation 已经
`PASS / VALID`，所以现在可以讨论 Phase B；该结果不自动授权 Bonn
真实来源执行。

Phase B 的下一唯一候选不是算法评价，而是：

```text
BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0
```

它只允许在未来另获明确授权后，核验官方来源权威，并在不读取 RGB、depth、
pose 数值、support、residual、trace 或既有分数的前提下冻结有限 sequence /
window cohort。设计审查、metadata gate 和正式 Phase B audit 是三道分开的门。

## 绑定权威

- 当前入口：`docs/research/rcle/README.md`；
- R1.1：`D:\edge\BlindAssist_RCLE_Minimal-First_R1.1.md`，
  SHA-256 `C6BC9E0D8A1C9665B319CB8141362B098B94B742772E351C24BE33445CC3BCD2`；
- Observable Support design lock：
  `3fcc21e28ba84e18d10b1c236a9a0df167d2a6464ea5ebefcb52ce4395152bac`；
- development receipt：
  `93b4c9244e9ef3bd11e8ab3557bfda0ad6dd6cd324116dbd05898f71b5214e3c`；
- sealed-validation receipt：
  `d10afb25cbe6bd8104b842adbd128b229804d5bdda0e8ff03d3954386806365c`。

上述 synthetic evidence 只开放 Phase B 的独立决策讨论，不提供真实来源有效性。

## 旧 Bonn 证据的永久角色

旧 Looming/Bonn 现场已经被观察或曾进入冻结选择，只能作为
`DISCOVERY_DIAGNOSTIC`。完整排除集由
`RCLE_PHASE_B_BONN_HISTORICAL_EXCLUSION_MANIFEST_2026-07-26.json`
绑定，共 `9` 条：

- 三条 prior-inspected、两条 discovery、两条 validation 和两条 sealed-holdout
  sequence 进入新 formal cohort 的权限均为 `DENY`；
- 原 canonical `3×3 / 500ms` truth 为 `18/18 abstain`；
- 后续 `503`-pair 评价把 global-image q90 signal 与 central-ROI q05 truth
  proxy 连接，空间单元不一致，保持 quarantined；
- 旧 base/oracle/full-6DoF traces、support、residual、评价分数不得参与新 cohort
  选择、阈值设置、准入或结论。

这些负边界可以用于设计防火墙，但不能升级为 Phase B 主结果，也不能因为文件已在
本地就绕过新的 authority/admission gate。

## 唯一 metadata gate

若未来单独授权该 gate，只能按以下顺序执行：

1. 先锁定代码、环境、receipt schema、官方页面/说明文件 identity 和输出位置；
2. 只读取官方发布的 dataset/sequence 级 metadata、文件清单、命名和格式说明；
3. 在任何 payload 解码前冻结有限 cohort、每条 sequence 的预期证据角色、
   时间窗生成规则和完整分母；
4. 生成 hash-bound receipt，并独立复算；
5. 任一来源权威、身份、metadata 完整性、旧观察防火墙或分母门失败，立即
   `HOLD_NOT_EVALUABLE`。

metadata gate 不运行 RCLE，不产生 raw/compensated/proxy 指标，也不决定
Kill Gate B。

## cohort 冻结约束

- 只允许 Bonn；不得同时搜索 REveL、JRDB、THÖR-MAGNI 或自采数据；
- 官方 universe 必须严格等于绑定页面 SHA-256
  `2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186`
  解析出的 `26` 个唯一 sequence；identity 或数量变化即 `HOLD_NOT_EVALUABLE`；
- 先排除历史 manifest 的 `9` 条，再保留官方 display size `<=550 MB` 的条目；
  对剩余条目按
  `SHA256("rcle-phase-b-bonn-entry-r1\t" + sequence_id)` 升序排列，固定选择前
  `6` 条作为唯一 formal admission cohort；
- metadata selection denominator 是官方全部 `26` 条，每条必须有
  included/excluded、排除原因和 rank；少于 `6` 条 eligible 时立即关闭候选；
- 选择的 `6` 条不预判 rotation/approach 角色；未来每个 endpoint 至少需要
  `2` 条可评价 sequence，否则该 endpoint 为 `NOT_EVALUABLE`；
- payload 读取前只冻结 sequence denominator 和 window 生成规则，不伪称已经
  知道 payload-derived window 数量；
- 若未来另获 payload-inventory 授权，对每条已选 sequence 一次性取
  RGB/depth/pose 的公共时间覆盖
  `[max(first timestamps), min(last timestamps)]`，从左端开始切连续、不重叠、
  不足 `10s` 丢弃的半开窗口 `[t0+10k,t0+10(k+1))`；这一次物化的全部窗口
  就是完整 window denominator，零窗口 sequence 保留且不得替换；
- sequence/window 不足时保留 `NOT_EVALUABLE`，不得替换失败单位；
- 最多连续三个工作日做 metadata-only authority 搜索；仍不能冻结合格 cohort
  时关闭本候选，不漫游式找数据；
- 不得以视频观感补写 pure rotation、approach、static surface 或 mixed-motion
  truth。

## 未来 formal Phase B 的独立授权边界

只有 metadata gate `PASS / VALID` 后，才可另立 Phase B implementation +
execution 设计。该未来设计至少必须结果前锁定：

- source-native pose/depth/静态模型各自可支持的 claim；
- raw、rotation-compensated、simple scale proxy 三方法的精确定义；
- sequence/trial 单位、时间对齐、空间共同单元和 abstention；
- rotation leakage、closing error、RSR/CRR、runtime、support rate；
- 多 sequence 同方向的数量门、完整分母、置信区间和失败语义；
- payload 只读顺序、oracle firewall、receipt schema、输出位置和运行次数。

即使 formal Phase B 最终通过，也不自动开放增强阶段、Replay Demo、Android、
人体、安全或生产路径。

## 当前禁止

- 下载、解压或读取 Bonn payload；
- 读取本地旧 Bonn RGB、depth、pose、map、trace、support、residual 或分数来选 cohort；
- 实现或运行 Bonn adapter、RCLE evaluator 或 proxy；
- 运行 Phase B、Kill Gate B 或 Replay Demo；
- 改 synthetic candidate、门、阈值或 receipt；
- 接触 REveL、JRDB、THÖR-MAGNI、自采、Android、人体、安全或生产工作。

## 下一授权句式

若设计锁通过独立只读审查，下一步精确授权可以是：

> 授权按 Phase B Bonn 入口设计锁实现唯一
> `BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0`，只执行
> metadata-only authority/admission gate；不得读取或下载 Bonn payload，
> 不得运行 Phase B 指标、Replay、Android、人体、安全或生产路径。
