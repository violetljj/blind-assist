# RCLE Phase B 真实 positive approach 数据角色准入 R0 预注册

状态：`PREREGISTERED_BEFORE_SOURCE_ACCESS / EXECUTION_NOT_STARTED`

日期：2026-07-27

机器合同：
[RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0_CONTRACT_2026-07-27.json](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0_CONTRACT_2026-07-27.json)

## 结论与边界

本阶段只补真实 positive approach 数据角色，不实现、不导入、不运行 RCLE RGB
algorithm。唯一候选冻结为 `EVIMO2 v2 / Flea3 / sanity_ll`；TUM approach
prescreen 已失败，Bonn、TUM `fr2/rpy` 和 Phase A 已烧掉或仅具 synthetic/
regression 权限，ETH3D、ICL、其他 EVIMO2 group/camera、镜像和替代来源均不属于
R0。

合法终态只有：

- `REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID`
- `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`

准入成功也只补齐数据角色。producer、独立 validator、immutable cache、progress
sidecar、guarded launcher、1 与 8/12 worker pilot、implementation lock 和独立
实现审查必须另立
`RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_IMPLEMENTATION_AND_PERFORMANCE_QUALIFICATION`
任务；正式 algorithm canary execution 仍需之后另行决定。

## 访问前冻结

source descriptor 的 canonical JSON 与 SHA-256 已写入机器合同。正式访问顺序为：

1. 只核验本合同和既有本地上游文件；
2. 以 exclusive create 建立并 `fsync` claim，claim 同时绑定 contract SHA 与
   source descriptor SHA；
3. claim 成功后才允许 GET official index 或触及任何 EVIMO2 candidate path；
4. 第一次完整 official response/payload 的 SHA-256 成为本 evidence version
   的 content identity，不因结果、大小或格式不利而换源。

官方页面未提供可在访问前独立验证的 payload checksum，因此本预注册不伪造
archive hash。访问前冻结的是 source-selection authority、canonical source
descriptor 及其 hash；取得后第一次完整 payload hash 自动成为不可替换身份。

## 唯一几何规则

只读取 source-native metadata、timestamp、calibration、camera pose、generated
depth 或静态场景几何。禁止解码 RGB、读取 event image、读取或推断 RCLE algorithm
output。

每条 source-native sequence 从首个共同 pose/depth timestamp 开始，建立固定、
非重叠、半开 `10.000 s` 窗；不滑窗、不按分数排序、不重心化、不截取、不换窗。
对连续 depth timestamp 使用 PB-H1 已解析校准的 `R·X` 对 `R·X+t`：

```text
signed radial expansion = log(rho(project(RX+t)) / rho(project(RX))) / dt
parallax = angle(unit(RX), unit(RX+t)) / dt
```

窗口需同时满足：

- candidate-pair coverage `>= 0.80`；
- evaluable pair `>= 8`；
- pair signed radial median 的窗口中位数 `>= 0.05 s^-1`；
- pair positive-fraction 的窗口中位数 `>= 0.75`。

所有固定窗都报告；角色内容是满足全部冻结门的完整窗口集合。全组没有通过窗即
HOLD，不启动另一个来源、另一个 group、滑动窗或阈值修订。

## Identity、independence 与 reuse

准入内容由 official payload SHA、source-native sequence ID 和 exact half-open
timestamp bounds 共同标识。独立组固定为
`EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY`，其 ancestry 与 TUM、Bonn、Phase A
不同，因此可承担独立真实 canary approach role。

本次访问同时把全部 EVIMO2 v2 Flea3 `sanity_ll`、准入窗及同 capture-family
derivative 永久排除在未来 confirmation partition 之外。成功内容只能作为
`GEOMETRY_SELECTED_REAL_APPROACH_CANARY_ONLY`，可保留为 canary/regression，
不得升级为 confirmation。
