# TARO O0M analytic runtime

状态：`IMPLEMENTATION_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN / EXECUTION_NOT_AUTHORIZED`

## 稳定 Interface

- `o0m_mechanics.py`：独立 NumPy SVD、factorial patch 与 action filter；
- `run_o0m_canary.py`：只接受已提交 one-shot execution lock，并原子占用独占 evidence root；
- `test_o0m_mechanics.py`：只使用 `impl_unit_*` 合成单元输入，不运行正式 O0M execution family。

## 输出

正式 runner 只向锁定的 `artifacts.local/evidence/taro/o0m-analytic-mechanics-r0/` 写入
`result.json`、`records.jsonl`、`execution-receipt.json` 与 `manifest.json`。目录一旦创建即消费
one-shot，不覆盖、不删除、不重跑。

## 安全边界

- identifiability 输入是预去重、预白化的解析矩阵，不证明真实 whitening/dedup pipeline；
- factorial solver 不接收 `truth_clearance_m` 或 expected records；truth 只由 runner verifier 使用；
- 不读真实数据、B1 outcome、网络、GPU 或设备；
- O0M PASS 只能建立 synthetic analytic mechanics，不能建立真实 factor causal headroom。

## 停止条件

没有已提交 one-shot execution lock、任一 binding 不匹配、evidence root 已存在、资源越界或任一
G01–G10 失败时立即 fail-closed。默认 App、产品与 safety authority 始终为 false。
