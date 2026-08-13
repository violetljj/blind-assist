# DepthART-S D3R3 Phase-B source-coverage transport recovery

状态：`D3R3_PHASE_B_SOURCE_COVERAGE_RECOVERY_HEAD_PROTOCOL_FROZEN_MEDIA_UNOPENED`

## 为什么另立 D3R3

D3R2 r0 在 exact-64 coverage-only census 的第 45 个请求
`44796744/lowres_depth.zip` 上收到 HTTP 200，但实际流式正文长度与冻结
`Content-Length` 不一致。该 root、44 个已保留 bodies、checkpoints、failure 和
temporary marker 均保持不可变；D3R3 不恢复、修补或复用它们。

D3R3 只继承 hash-frozen 的 exact-32 identity 顺序、每身份 exact-300 stems、
9600-stem plan、64 个 URL 及 ARKitScenes source/license 事实。当前没有科学结果，
`scientific_terminal=null`。

## 当前冻结的最小恢复门

第一步不是重下正文，而是对全部 64 个 URL 建立新的 HEAD-only snapshot。只检查第
45 个资产会让其余 63 个资产仍停留在旧时间点，不能形成同版本 source snapshot。

该 HEAD 门固定：

- exact 32 identities × `lowres_depth.zip/confidence.zip`，严格 64 个 URL；
- 每次请求只能是 HEAD，禁止 redirect、GET、Range 和 response body；
- 每行必须 HTTP 200、final URL 不变、正整数 Content-Length、非空 ETag 与
  Last-Modified；
- transient transport、408、429、5xx 最多三次（含首试），其他错误不重试；
- 逐字段记录相对 D3R1 旧 HEAD 的 header drift；有效的新 HEAD 是 D3R3 的未来
  authority，旧 HEAD 只作对照；
- 独立 validator 重算请求计划、attempt history、availability、header drift 和
  terminal；它不导入 producer 的判定函数。

当前只冻结 scope、协议、producer、validator 和 synthetic fixtures；正式 activation
与 evidence root 都不存在，也没有发送 HEAD。

## HEAD 通过后仍不能做什么

HEAD PASS 只能进入
`EXPLICIT_D3R3_PHASE_B_EXACT64_COVERAGE_ONLY_CENSUS_PROTOCOL_REGISTRATION`。
届时另行冻结 body/census 协议，绑定 fresh headers；随后还需独立 activation 才可
全 64 重新 GET。D3R2 的 44 个 bodies 不得复用。

未来 body 协议应把 fresh-header-matched 的 premature EOF 分类为
`TRANSIENT_BODY_SHORT_READ`，允许同一资产从 byte zero 做最多三次完整尝试；
仍禁止 Range/partial resume，并累计包括失败 short read 在内的所有 transport body
bytes。该规则当前只是设计约束，不是执行授权。

## 当前唯一下一门

`EXPLICIT_D3R3_PHASE_B_EXACT64_FRESH_HEAD_ONLY_PREFLIGHT_ACTIVATION`

它必须由新的明确指令激活。当前禁止 HEAD/GET/Range、archive/member、decode、
truth、selection、role、training、Development/R2 outcome、performance、默认 App、
production 与 safety claim。

机器协议：
[JSON](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_RECOVERY_PROTOCOL_2026-08-13.json)

source-scope：
[receipt](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_SCOPE_RECEIPT_2026-08-13.json)

