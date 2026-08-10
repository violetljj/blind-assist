# TARO O0R source-and-adapter contract lock result

终态：`TARO_O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN`

## 结果

新的 ARKitScenes source-and-adapter 合同通过静态验证。validator 重算了 14 个 bound files、
Git commit `1cc126e7` 上 100 个历史 ARKitScenes identity 的 exclusion snapshot，以及 metadata-only
SHA 排序得到的 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` roster；24 个 visit/video 无交叉，
21/21 mutation tests PASS，四个未来 TARO root 均不存在，历史 O0M root 保持只读。

合同已经在 outcome 前冻结：

- FARO-only scale/support/boundary 与 body/path query truth；
- exact decimal timestamp、K、pose、gravity 和 parent/session/site receipt；
- 只用 8 个 fit parent 的真实 cross-source residual 拟合 uncertainty，不允许 constant sigma；
- 16 个 eval candidate 的 truth-only admission、至少 12 个 evaluable parent、clear/occupied 各至少
  6 parent，以及 undefined denominator fail-closed；
- DepthART-S baseline identity、八臂 copy-patch-reduce injection、primary estimand、`0.02 m` minimum
  effect、false-clear/known-coverage non-inferiority、bootstrap、预算和 failure scope；
- 与已消费 O0M evidence 的独立 namespace。

提交前的独立接口审查又关闭了五个实现缝：truth-only SCALE 改为 model-free FARO absolute
reference，candidate-relative correction 延后到已签署 truth-only result 之后；pose watermark 包含实际
右 bracket；每物理帧固定 base receipt + 9 个 query-bound receipts；source-specific receipt 明确不冒充
缺 covariance/tracks/camera-body transform 的完整 P0 receipt；旧三带 reader/reducer 仅作 primitive/reference，
后续实现必须使用新的 exact-timestamp TARO query reducer。registration、virtual ground-frame capsule、
boundary sign/completeness、uncertainty fallback/hash 与三个 admission denominator 也已机器冻结。

## 没有发生的事

本轮没有下载或打开 24 个 selected source body，没有物化 truth/uncertainty，没有运行 DepthART，
没有执行 factorial、训练、G0/G1/A0/A1/J0、Android/QNN/HTP，也没有创建 scientific artifact。
所以这里的 PASS 是合同/接口 PASS，不是 O0R 科学 PASS。

## 唯一 successor

`TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK`

下一步只允许实现并静态测试冻结的 receipt、truth、uncertainty 和 factor-injection adapter；在实现
hash 与独立 truth-only one-shot preflight lock 提交前，source download、truth materialization、
DepthART inference 和 O0R execution 仍为 false。

## Claim ceiling

只建立 validated pre-outcome source/role/adapter contract。不建立完整 P0 FrameReceipt、独立 metric
anchor 或真实 camera-body mount，也不证明 factor causal headroom、GaugeFix、PARA、穿戴式主动观察、
设备、产品或安全有效性；默认 App 与其他路线不变。
