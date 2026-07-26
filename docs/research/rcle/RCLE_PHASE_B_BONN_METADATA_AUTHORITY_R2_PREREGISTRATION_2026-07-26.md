# RCLE Phase B Bonn Metadata Authority R2 预注册

状态：`DESIGN_FROZEN / EXECUTION_AUTHORIZED_IF_REVIEW_PASS`

日期：2026-07-26

## 唯一修复

R2 固定采用 R0/R1 已见但 outcome-free 的同一
`26 / 9 / 6 / 513b770d…ae86e` 内容，不重选 cohort。唯一修复是 claim 时序：

1. formal runner 解析无路径 override 的 CLI；
2. 仅用代码常量与 repo root 推导 canonical R2 `run_claim.json`；
3. implementation lock 冻结前，由独立 setup 一次性预创建 canonical output
   directory 并写入 hash-bound `directory_setup.json`；formal runner 不得创建、
   检查或读取该 marker；
4. 第一项 formal-runner 文件读写必须是 `O_CREAT | O_EXCL` 创建最小 claim；
5. claim 后才允许读取/哈希 implementation lock、setup marker、R0/R1 receipt、official HTML、
   historical manifest、controls 或 environment；
6. claim 后任一失败均保留 claim并永久禁止 R2 重跑；
7. validate-only 不创建 claim，只读 canonical R2 输出。

Pre-claim 禁止 `mkdir/exists/stat/open/read/hash/listdir/glob` 及网络访问；只允许纯
路径计算和 exclusive claim。

## 固定 identity

- R0 receipt：`4386bbe3…f1764`
- R1 diagnostic receipt：`c2efac24…9ed92`
- official page：`2bd8df16…0c186`
- historical manifest：`f02bd9f1…b1cc6`
- cohort：`513b770d…ae86e`

R1 永久保持 diagnostic，不作为 R2 的执行权威；其 receipt 只用于证明 adoption
内容没有发生漂移。

## 终态

- PASS：
  `CANONICAL_METADATA_AUTHORITY_R2_PASS_FORMAL_PHASE_B_B0_READY`
- FAIL：
  `CLOSE_CANONICAL_METADATA_AUTHORITY_R2_NO_RERUN`

R2 不读取 Bonn payload，不授权 metrics。PASS 后只使已冻结的 Phase B B0
acquisition/timestamp-inventory 协议具备执行入口。
