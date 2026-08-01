# HFTF Stage C D4-M0 metadata census invalid result

## 结论

D4-M0 已按[冻结合同](HFTF_STAGE_C_D4_M0_METADATA_CENSUS_EXECUTION_CONTRACT_2026-08-02.md)关闭为
`D4_M0_FRESH_METADATA_RECRUITABILITY_POOL_INVALID_STOP`。正式 one-shot 在
[attempt](../../../artifacts.local/evidence/hftf/stage-c-d4-m0-metadata-census-20260802/attempt.json) /
[preflight](../../../artifacts.local/evidence/hftf/stage-c-d4-m0-metadata-census-20260802/preflight.json)
后开始；[failure](../../../artifacts.local/evidence/hftf/stage-c-d4-m0-metadata-census-20260802/failure.json)
以 `OSError: [Errno 22] Invalid argument` 关闭。

这不是 5 Hz pool 不足，也不是 opportunity/HFTF 负结果；它只是 metadata census
未完成的执行无效终态。同一 canonical root、同一 1442-candidate census 或“修 transport
后再来一次”均未获授权。

## 已可靠建立的事实

- 执行前 `HEAD == origin/master == 72af4c7...`，formal Git/hash gate 通过；
- attempt 与 preflight 在首网前 durable；
- preflight 只读 exact 40 个历史 slot attempt，不读 sealed payload/selector/truth；
- 40 个历史 session IDs 全在 frozen global-124 exclusion union；
- canonical failure 已按合同被 terminal validator 接受。

文件系统时间与“wrapper timeout 后继续监控原 PID”的过程仅是**未绑定的 operator
observation**：没有 process-log/PID-monitor receipt，不能作为 canonical evidence，
也不参与 terminal 或 claim。

收据 SHA-256：

- attempt:
  `7ba7f6a6bc9404fbe43dfee2955ad853929b32a7d7a310dcba4a38ccf404feb8`
- preflight:
  `52735837a65f52603c31c4a3e6a2d76986d63e4cebb322904aadea34182efeb4`
- failure:
  `b9fb61cd33cd820113b246aaf9cf36ac58379dc37a916b5a03ff47fbafba96f5`

## 没有产生的证据

`census.json`、`pool.json`、allocation attempt、seed 和 result 全部不存在。因此：

- 没有完整 1560-row ledger；
- 没有 5 Hz/20 Hz eligible count 或 pool；
- 没有随机 rank、ecology/effect reserve IDs；
- 没有打开 fresh pose/media bytes、support、truth、effect 或 sealed payload；
- 没有 ecology、effect、student、主线/App/Android、生产或 safety 权限。

若继续这条潜力支线，必须另立新协议与新 source population；不能把本次已尝试的
1442 parents 包装成 fresh，也不能用 transport patch 把 R0 复活。

机器结果 SHA-256：
`bba56892cd579b2e278705070ad6f42cbb6db1bc1264ec99de3132f9d888c993`。
[机器结果 JSON](HFTF_STAGE_C_D4_M0_METADATA_CENSUS_INVALID_RESULT_2026-08-02.json)。
