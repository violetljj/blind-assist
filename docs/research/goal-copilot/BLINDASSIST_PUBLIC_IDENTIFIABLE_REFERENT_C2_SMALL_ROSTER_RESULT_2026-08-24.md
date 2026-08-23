# Public Identifiable Referent C2 small-roster result (2026-08-24)

终态：`SMALL_ROSTER_MATERIALIZABLE / 7_SOURCE_DISJOINT_EPISODES / REFERENCE_IMAGE_INSTANCE_UNIQUE_ONLY / NO_BASELINE / NO_ALGORITHM`

## 结论

按预注册的 [`C2 small-roster protocol V1`](BLINDASSIST_PUBLIC_IDENTIFIABLE_REFERENT_C2_SMALL_ROSTER_PROTOCOL_V1_2026-08-24.md)
执行了一次 metadata freeze 和一次 materialization。SUN3D 官方 pose-corrected registry 中，排除已消费的
`hotel_umd/maryland_hotel3` 后，固定的 7 个 source 全部产生了一个合格 episode；没有补源、换源、同 source 切第二例、
看 RGB 后换 target/frame 或调用任何模型。

因此 C2 唯一科学问题得到肯定答案：可以实际构造一个 5--8 例范围内、公开 reference image + public target region
唯一指定 referent、私有 physical identity 预先锁定、later truth 可绑定的 source-disjoint 小 roster。

这不回答“系统能否重新找到同一个实体”。Passive baseline、detector、matcher、Active Referent Search、FSM、control、
产品能力均未运行、未获授权。

## 冻结与 materialization 结果

| 项目 | 结果 |
| --- | ---: |
| 固定 metadata sources | 7 |
| 合格 / source-disjoint episodes | 7 / 7 |
| metadata GET | 15 / 15 budget |
| freeze 前 image GET | 0 |
| `REFERENCE_IMAGE_INSTANCE + UNIQUE` contracts | 7 / 7 |
| reference images | 7 |
| later observations | 21 |
| frozen real-viewpoint gate | 21 / 21 |
| V1 truth binding / evaluator-primary rows | 21 / 21 |
| 含 same-class distractor 的 episodes | 6 / 7（diagnostic，非成功门） |
| image GET / unique image SHA-256 | 28 / 28 |
| image payload | 6,481,797 bytes / 20,000,000 budget |
| provider / teacher / detector / matcher / baseline calls | 0 / 0 / 0 / 0 / 0 |

下载 journal 共 56 个 dispatch/complete events。前 14 个 events 恰为 7 张 reference 的 dispatch/complete；第一笔
later-image dispatch 发生在第 15 个 event。`identity-lock-barrier.json` 此时记录
`reference_image_gets=7 / later_image_gets=0`，并绑定全部 7 份 public contract 与 private identity lock hashes。
公开 manifest 对冻结 roster 中全部 private physical instance IDs 的命中数为 0。

## Evidence anchors

- frozen roster body SHA-256：`5e22b3f0cb9ef3f4fc0dc1ecdfa2fa3f7679ced79da0e74e07101f3717b67196`；
- frozen roster file SHA-256：`c532111e1f43c6faa84d8128d3b1fe0dd75127c514ccbaa1bfe8318f0fdc8a86`；
- identity-lock barrier body SHA-256：`8fa7d8f2921d350c5dba5a7b32692dd026d75bb9ec4f8cbfa392d911229c26ce`；
- public manifest body SHA-256：`5f0785a75116f3e8e3c4ec3e1e78c54385211a876d1eea693005008c6d36af8a`；
- private manifest body SHA-256：`cc5a4d15033f05dbf5869b28bcdc945621a5503299016c257c0ee0778ec76f7f`；
- final report body SHA-256：`a61eebf527f35a7b3b72e2b475134ae9641f07aca9993f12ba7b4642477f9d1b`；
- final report file SHA-256：`5c46d252d445c9d0ab485fc95172939dba95a1616bd2e240afa19bdf041867ca`。

本地证据根：`artifacts.local/evidence/public-identifiable-referent-c2-v1/`。其中 roster、reference/later images、
public/private receipts、truth audits、journal、barrier、manifests 与 final report 均保留；它是 ignored evidence，
不进入 Git。

## 研究边界与下一步

Claim ceiling：
`PUBLIC_IDENTIFIABLE_REFERENCE_IMAGE_SMALL_ROSTER_MATERIALIZATION_ONLY_NO_IDENTITY_BASELINE_ALGORITHM_NAVIGATION_CONTROL_SAFETY_OR_PRODUCT_CLAIM`。

C2 到此关闭，不继续扩建 contract，也不自动执行 C3。若另行授权，下一版本只能是一个极简单的 passive baseline，
读取 provider-public reference evidence 与 later images，在 evaluator-private truth 上分别报告 visibility、proposal
availability、same-instance identity、wrong-instance commit 与 honest abstention；其观察结果再决定是否需要 Active
Referent Search、proposal repair 或 identity verification。
