# HFTF Stage C D4-M0 metadata census execution contract

## 结论

D4-M0 已具备可提交审计的 one-shot execution contract，但在合同、planner、tests
提交推送且独立终审为 `CLEAR` 前仍不得正式执行。M0 只建立 fresh 5 Hz
metadata-eligible pool 与机械随机分配，不产生 opportunity、effect 或 safety
证据。

实现前复核纠正了原设计中的集合投影错误：全局排除权威仍为 124 parents、8060 bytes、
SHA-256
`156bf17c54ecfba41f181a12df209aecc56b3c9a6a85f27b2db2f340737252f2`；
但其中 6 个是 official-test parents，不在 1560-ID official-train split。因此正式
ledger 必须是：

```text
1560 train rows = 118 in-split exclusions + 1442 candidate attempts
```

不能为凑旧的 1436 数字再跳过 6 个合法 train parents。

## 内容防火墙

每个非排除 candidate 只允许读取 description bytes、pose object metadata，以及
normalized exact-13 mask/depth object listings。RGB listing/bytes、pose CSV bytes、
mask/depth bytes、support、truth、clearance、effect 与 sealed payload全部禁止。
118 个 in-split exclusions 不得发起 candidate metadata 请求；ineligible/404 是闭合
ledger row，不补位、不重跑。

## Durable one-shot 状态

正式路径固定为
`artifacts.local/evidence/hftf/stage-c-d4-m0-metadata-census-20260802`，状态顺序固定：

1. exclusive attempt 在首个网络请求前 file-fsync、关闭并逐字节重开验证；
2. 完成 1560-row census 并以相同屏障持久化；
3. 持久化完整 5 Hz pool manifest；
4. `N<64` 时不创建 allocation attempt、不生成 seed，直接不足终态；
5. 否则先持久化绑定 pool hash 的 allocation attempt；
6. 仅调用一次 `secrets.token_bytes(32)`，再持久化 seed receipt；
7. 以
   `SHA256("HFTF_D4_R0_ALLOC|" || seed_bytes || "|" || lowercase_session_id)`
   唯一 digest 升序排名；collision 直接 invalid；
8. 机械代入 `C=min(N,128)`、`n=floor(3C/8)`、`B=C-n`，持久化
   ecology/effect/unassigned 三个互斥集合。

Windows 不提供受支持的 directory-fsync；合同没有伪称这一能力。可验证的 Windows
屏障是 exclusive create、文件 `fsync`、关闭、再以 exact bytes 重开校验。任何
partial/unknown canonical root 的后续调用只会写 INVALID failure，不联网、不续跑、
不生成或重抽 seed。

## 验证和权限

focused tests 为 `21/21`，HFTF full suite 为 `413/413`。其中回归证明 local drift
审计不会打开 sealed payload、drift 会在首网前 durable INVALID、transport exhaustion
不会伪装为 metadata ineligible，且 corrupt/unknown result 不会被承认为终态。正式 CLI 还要求合同、设计、
planner、test 与绑定 parents 全部 tracked、clean、hash 精确，并满足
`HEAD == origin/master`；其中 `artifacts.local` parents 按项目规则不进 Git，
以 exact path/SHA/schema/terminal 校验。通过 M0 也只授权另冻 ecology execution contract；不自动
授权 fresh content、ecology、effect、student、研究主线、App/Android、生产或 safety。
机器合同 SHA-256 为
`21a6de0e16e65998318aa83b549c3467eb9fe2b59193faa1fa44d72d1d891759`。
