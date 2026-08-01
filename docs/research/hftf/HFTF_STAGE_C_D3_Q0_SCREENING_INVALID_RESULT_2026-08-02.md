# HFTF Stage C D3-Q0 screening invalid result

终态：

`D3_QUALIFICATION_INVALID_STOP`

## 结论

D3-Q0 在冻结 roster slot 1 上只启动了一次正式媒体/资格进程，但结果不能进入
screening cohort。runner 已 durable 写出 attempt、content index、sealed payload 与
selector；随后 closed-schema 自检发现 selector 多了禁止的顶层
`slot_attempt_sha256`。同一个 attempt hash 已正确存在于允许的
`source_authority_and_content_hashes` 内，因此这是 runner 的重复字段实现缺陷，不是
数据或资格门结果。

没有修改 validator 来接纳已见结果，也没有重写 selector/payload、重启媒体进程、
换源或继续 slot 2。恢复入口只重扫 receipt 并 durable 封存
`screening_invalid.json`，没有重开媒体或 sealed payload。

## 执行事实

- execution contract commit：
  `306477105db033dbb805fc78bd8567c2afb29b34`
- execution contract SHA-256：
  `84f24a72c4640ca3ba66388ed9ec75a68aa55270c5e369b2b072a7b4d65354eb`
- 正式执行前 `HEAD == origin/master` 且 `verify_git=True`；
- slot 1 只下载 1 个 pose CSV、11 个 mask、11 个 depth，RGB 为 0；
- 正式媒体/资格进程只启动一次；
- selector 的 forensic terminal 是
  `D3_Q0_SLOT_REFERENCE_SUPPORT_OPPORTUNITY_NOT_QUALIFIED`，但 selector schema
  非法，因此该 terminal 和所有 strata 都不具备 cohort admission authority；
- selection、budget terminal、aggregate attempt、slot failure 均不存在；
- slot 1 已打开 support/truth，因而永久 burned。

关键 durable hashes：

- screening attempt：
  `137d0fa065c2eabd61fdc2ba158b12d9f586c1021fe2b0e64a292faf5492f364`
- slot attempt：
  `bff9cc469a1b9571fa9e858eafe853e646fba8a935476bf9e2e225b7c08e44f8`
- content index：
  `7df2d5fbeab7483235f38b8fd9f2fa50007eab8c909ba55fa529a620b2610f6a`
- sealed payload：
  `7a1271ffa876df453df38ea52ba3db4c14044631ef9dc70e44023ea5433d55ed`
- invalid selector：
  `cbad78e83d3b3aca80a2a9faaa6d14bde2151ae08e10fc9e2f922d99a1814865`
- screening invalid terminal：
  `e1975e896b5d6a26f8a28ee7ee29b5a9d1d3f4cc53b0a183c3dd0aec658e962d`

## 允许的后继

本终态只允许冻结一个全新的 schema-only Q0.1 execution contract：

- 只删除重复且被禁止的 selector 顶层字段；
- slot 1 永久 burned，不重开、不迁移为合格/不合格 receipt；
- remaining order 必须从原 roster slot 2 开始，最多 39 个 slots；
- qualification axes、`.10/5/20` 门、252 denominator、UNKNOWN 规则、effect gates、
  source order 与 no-replacement/no-expansion 政策全部不变；
- 必须使用新 contract、新 canonical root，并在任何 slot 2 media 前重新提交推送、
  `verify_git=True` 和独立审计。

当前不授权 future-blind prediction、sealed effect、RGB student、reserved official
test、研究主线或 App 变更、生产或安全声明。
