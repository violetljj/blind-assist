# RCLE Phase B RGB algorithm canary R0 性能与 preflight 规范

状态：`DESIGN_FROZEN / PERFORMANCE_NOT_QUALIFIED / EXECUTION_NOT_AUTHORIZED`

日期：2026-07-27

## 适用边界

本规范只定义未来 implementation task 在创建任何 formal claim 前必须满足的 archive mechanics、性能、进度和独立复算门。本文没有实现或运行算法，没有创建 implementation lock、activation、claim、output 或 failure receipt，也没有读取 RCLE RGB algorithm outcome。

geometry canary R0 在约 `2.05 GB` gzip TGZ 上逐 pair 重复 `extractfile/read`，producer 与 validator 合计产生约 `2156.646 GiB` 逻辑读取。该事实只证明旧访问路径不合格，不改变 geometry interface parity 的既有结论。

## Immutable cache

未来任务必须在 claim 前按 TGZ archive header order 单次顺序物化 cache。archive
ordinal 是规范顺序；member path 只要求规范化且唯一，不要求词典序，因此不会为了
排序在 gzip stream 上做第二次回扫。禁止在正式 pair loop 中打开 gzip TGZ 或随机
读取 member。cache root 必须位于 `artifacts.local/`，先写入新目录，完整验证后
原子发布为只读版本；存在同名目录时 fail closed，不覆盖、不增量修补。

cache manifest 至少记录：

| 字段 | 语义 |
| --- | --- |
| `schema_version` | 固定 cache manifest schema |
| `source_path` | 仓库约束内的 canonical source 引用 |
| `source_size_bytes` | TGZ 精确字节数 |
| `source_sha256` | `3a35b799…62b51f` |
| `members[].archive_ordinal` | TGZ header 的零起始顺序，严格连续递增 |
| `members[].member_path` | 规范化且唯一的相对 member path |
| `members[].size_bytes` | 解压 member 的精确字节数 |
| `members[].sha256` | 解压 member 内容 SHA-256 |
| `member_count` | manifest 中 member 数 |
| `total_materialized_bytes` | 全部 member size 之和 |
| `manifest_sha256` | 对移除本字段后的 canonical JSON 计算 SHA-256 |
| `created_at_utc` | 物化完成时间 |
| `status` | 仅允许 `VERIFIED_IMMUTABLE` |

发布前必须全量复算 source hash、archive ordinal、member path 唯一性、size、
member hash、member count 与 manifest hash。producer 与独立 validator可以共享
该已验证 cache；共享的是 immutable bytes，不是 scientific summary、pair ledger、
aggregate 或 producer 代码。任一 cache member 或 manifest 漂移使本 evidence
version `INVALID`，不得创建 claim。

## Bounded real-mechanics pilot

pilot 必须使用与正式任务相同的 source TGZ、顺序物化代码、cache layout、image decode、pair computation路径、进度 sidecar writer 和 process scheduling。只允许固定的非科学计分子集：每个 planned window 的开头连续 `32` pair；window `4` 仍只作 abstention/interface stress。pilot 输出标记 `PERFORMANCE_PILOT_ONLY`，永远不得进入科学 aggregate。

必须比较 `1` worker 与 `8` worker；实现任务可用 `12` worker替代或补充 `8` worker，但必须至少有一个多 worker 配置。OpenCV/BLAS nested threads 固定为 `1`。两种配置必须对相同 pilot pair 产生严格相同的有序 identity、schema、abstention、IEEE-754 hex 数值和 aggregate；任何差异返回 `PERFORMANCE_NOT_QUALIFIED`。

每个配置记录：

- wall time、CPU time 与 `core_equivalent = cpu_time / wall_time`；
- source/cache read bytes、write bytes 与 TGZ sequential-read count；
- peak RSS、available RAM at start、worker count；
- completed pair count、pair/s；
- 基于每窗 pair 数和独立 validator 再计算成本的 projected wall time；
- `projected_wall_time_s`、`max_wall_time_s` 及其推导；
- 两个以上来自真实运行、时间戳严格递增且 `completed` 增长的 progress samples。

## Qualification gates

| ID | Unit / 判据 | Rationale | Calibration source | Sensitivity plan | Revision policy |
| --- | --- | --- | --- | --- | --- |
| `P-CACHE-SEQUENTIAL` | TGZ sequential passes，`EQ 1` | 消除已观测到的逐 pair gzip 回扫 | geometry canary R0 的 `2156.646 GiB` 逻辑读取 incident | 同时报告 source bytes 与 cache bytes；若计数器口径不稳定，保留操作级 trace | claim 前可新建设计版本；本 R0 不原地放宽 |
| `P-CACHE-IO-AMPLIFICATION` | materialization source-read / archive-size，`LTE 1.10` | 单次顺序读取允许至多 10% 计数开销 | TGZ 精确 size 与 OS I/O counter；不是科学 gate | 以 process I/O 和文件系统 counter 双口径报告 | 计数口径变化需新版本与独立 review |
| `P-WORKER-EQUIVALENCE` | mismatch count，`EQ 0` | 调度不得改变科学 bytes | 同一真实访问机制 pilot 的 1 与 8/12 worker 输出 | 同时做原顺序和逆完成顺序的 deterministic merge mutation | 任何 mismatch 先修实现并另立 implementation lock；不得容差回救 |
| `P-PROJECTED-WALL` | seconds，`LTE 3600`；hard max `5400` | 限制 one-shot 运行暴露与夜间不可观测时间 | 旧 R0 `~7620 s` incident；本任务给下一版本分配 60 min projection / 90 min hard stop | 同时报 45/60/90 min 情景；结论不随情景改变，只影响执行资格 | 只能在 claim 前以新设计版本修改，需说明资源预算 |
| `P-RAM` | bytes，`LTE min(16 GiB, 0.75 × start_available_ram)` | 防止 worker 扩张触发 swapping 或系统不稳定 | host compute policy 与 pilot peak RSS | 8/12 worker都记录；若 12 超门而 8 合格，选择 8 | claim 后不可改 worker 数；claim 前变更需新 preflight receipt |
| `P-PROGRESS` | valid samples，`GTE 2`；completed 严格增长 | PID 存活不能证明健康进展 | geometry R0 缺少 pair-level progress/ETA 的 incident | 以至少两个采样间隔复核 throughput 与 ETA 单调合理性 | 缺失或停滞即 `PERFORMANCE_NOT_QUALIFIED` |

门槛只决定执行资格，不是算法效果阈值。若 1 worker 已满足时间/RAM门，多 worker没有提速但严格等价，可选择 1 worker并记录原因；若所有配置未通过，返回 `PERFORMANCE_NOT_QUALIFIED`，不得创建 formal claim。

## Progress sidecar

sidecar 必须原子替换写入，schema 至少包含：

```json
{
  "phase": "CACHE_MATERIALIZATION|PRODUCER|INDEPENDENT_VALIDATOR|TERMINAL",
  "completed": 0,
  "total": 0,
  "throughput_per_s": 0.0,
  "eta_s": null,
  "last_progress_at": "RFC3339 UTC",
  "pid": 0,
  "input_sha256": "64 lowercase hex",
  "implementation_sha256": "64 lowercase hex",
  "status": "STARTING|RUNNING|VALIDATING|VALID|INVALID|FAILED"
}
```

`completed <= total`，running 状态必须有正 PID，input/implementation hash 必须与 preflight receipt 完全一致。正式 validator 必须拒绝缺字段、陈旧 implementation hash、倒退 completed、伪造 terminal 或少于两个真实 progress sample 的 evidence version。
相邻 `RUNNING` sample 的 `last_progress_at` 必须严格递增且间隔不超过 `120 s`；
未完成时 ETA 必须为有限非负数，phase/status 必须来自枚举，PID 必须为正数。`120 s`
是运行健康门，不是科学阈值；pilot 必须同时报告 `60/120/300 s` freshness
敏感性。修改 freshness 门只能发生在 claim 前的新版本中。

## Guarded launch

implementation task 只有在下列条件同时满足后，才可另行请求 formal execution authority：

1. 算法实现、独立 validator、cache materializer 与 launcher 形成新的 implementation lock；
2. bounded pilot 通过全部性能门并产生 `scripts/validate_host_research_preflight.py` 接受的 receipt；
3. data manifest 已补齐独立、真实、positive approach role，且未侵占 confirmation；
4. outcome firewall 证明 claim/output/failure 均未出现；
5. 独立 implementation review 通过。

未来长跑必须经 `scripts/run_guarded_host_research.ps1` 启动。底层 runner 不得直接调用；preflight 不合格、receipt hash 漂移或 progress contract 缺失均返回 `PERFORMANCE_NOT_QUALIFIED`，且不得创建 claim。
