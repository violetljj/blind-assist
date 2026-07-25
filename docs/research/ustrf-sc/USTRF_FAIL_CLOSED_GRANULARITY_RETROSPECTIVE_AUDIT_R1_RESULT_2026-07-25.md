# USTRF fail-closed 粒度回顾审计 R1（2026-07-25）

状态：`RETROSPECTIVE_COMPLETE / HISTORY_PRESERVED`

## 结论

近期记录里确有“把局部不可用扩大成整段不可用”的误杀，但不能反过来把所有旧失败都洗成通过。审计按结构完整性、claim dependency、局部缺失、support、方法性能与 authority gap 分开复核，旧 receipt 和当时终态均不改写。

| 历史节点 | R1 分类 | 未来处置 |
| --- | --- | --- |
| CrowdBot 首个 holdout，窗口内出现与目标无关的 unknown people 即整窗关闭 | 明确的 protocol-level false reject | unknown object 只让依赖该对象的 metric/unit abstain；不得抹掉窗口内其他可用证据 |
| CrowdBot replacement `0/2` | 部分过宽，但 selection pack 的不足仍真实 | 保留逐 metric evidence；不足的 metric `NOT_EVALUABLE`，不把 partial evidence 写成 selection pass |
| JRDB global-affine `11/31` | 不是数据缺失误杀，是当前 method/window 的 availability failure | 保留 `11/31`；failure scope 限于 frozen global-affine method/window，不扩写为 JRDB source 整体无用 |
| G0 `SOURCE_AUTHORITY_ABSENT` | 对目标 signal claim 的 authority failure 真实，但不应否定 RGB/time/membership 等其他证据 | 只关闭依赖 canonical transform/完整 geometry 的 claim；其他已验证角色继续保留 |
| JRDB P2 R0 的 29/1,350 个 3D-only | 明确的 claim-dependency false reject | R1 已修正：3D geometry/motion 保留；cross-modal identity 对 29 个 object-frame abstain |
| NavWareSet `14.0799s` raw time range 不一致 | 结构性 time-authority failure，不是普通缺失 | 依赖该时钟的 claim 关闭；只有能证明污染被严格定位时，才允许缩到更小范围 |

## 未来判定规则

1. 全局 `INVALID` 需要可复核的全局传播证据；“存在缺陷”本身不够。
2. 默认影响粒度为最小依赖单元：field → object → frame → pair → window → sequence → source → program。
3. 每个 claim 单独列 required/optional roles；optional role 缺失不得关闭不依赖它的 claim。
4. 每个 source-native denominator 必须守恒；不得用交集、静默过滤或删除 unknown unit 美化 coverage。
5. `AVAILABLE_WITH_DEGRADATION` 不是“啥都算通过”：必须同时报告 abstention、missingness cluster、bias risk 和 authority ceiling。
6. 方法表现失败、数据结构失败和 authority 缺口不得互相替代。一次 method failure 不等于 source failure；一次 partial availability 也不等于 selection 或人体安全证据。

控制标准见 [USTRF 弹性证据与降级标准 R1](USTRF_ELASTIC_EVIDENCE_AND_DEGRADATION_STANDARD_R1.md)。
