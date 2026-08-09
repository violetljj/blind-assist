# TARO O0M protocol lock

状态：`NON_EXECUTION_PROTOCOL_LOCK / IMPLEMENTATION_NOT_AUTHORIZED / EXECUTION_NOT_AUTHORIZED`

本 Module 只保存 O0M execution-family fixture 的静态 validator 与 mutation tests。当前没有
mechanics implementation、runner 或 scientific artifact。动态状态与唯一 successor 见
[`docs/research/taro/README.md`](../../../docs/research/taro/README.md)。

## 稳定 Interface

- `validate_taro_o0m_protocol.py`：重算 10 个 identifiability truth、80 条 factorial records、
  action filters、binding 与 authority；
- `test_validate_taro_o0m_protocol.py`：33 个 mutation tests；
- 当前不存在 `taro_o0m_runtime`、solver 或 runner。

## 输出

当前只向 stdout 输出静态验证 JSON，不写 artifact。未来 one-shot 只能写入冻结且事前不存在的
`artifacts.local/evidence/taro/o0m-analytic-mechanics-r0/`。

## 安全边界

- Protocol PASS 不等于 scientific PASS；
- 不读 real data、B1 outcome、device 或其他路线 artifact；
- VALUE_ONLY 不得改变 validity、sigma、provenance 或 common-support；
- synthetic mechanics 不得称为真实 factor causal headroom。
- factorial solver 只消费 `observed_base_mean_m` 与 patch delta；truth 只供 verifier 对照；
- deterministic budget halfwidth 不得冒充 Gaussian `1σ` 或 95% coverage。

## 停止条件

任一 binding、truth、record hash、body-motion filter 或 authority 漂移即关闭本协议版本。当前唯一
successor 只允许另锁 implementation；没有 one-shot execution lock 时不得运行 canary。
