# HFTF Stage C D5-S0A TartanGround catalog invalid result

## 结论

D5-S0A 已按[冻结合同](HFTF_STAGE_C_D5_S0A_TARTANGROUND_CATALOG_EXECUTION_CONTRACT_2026-08-02.md)
关闭为 `D5_S0A_TARTANGROUND_DIFF_CATALOG_INVALID_STOP`。唯一一次 formal CLI 已先
durable 写入
[attempt](../../../artifacts.local/evidence/hftf/stage-c-d5-s0a-tartanground-catalog-20260802/attempt.json)
与
[preflight](../../../artifacts.local/evidence/hftf/stage-c-d5-s0a-tartanground-catalog-20260802/preflight.json)，
随后以
[failure](../../../artifacts.local/evidence/hftf/stage-c-d5-s0a-tartanground-catalog-20260802/failure.json)
中的 `ValueError: Unexpected declared size format at row 978` 关闭。

这只是清单 size 语法合同与真实清单不一致的执行无效终态，不是目录容量不足，
也不是 TartanGround source feasibility、机会或 HFTF effect 的负结果。同一 canonical
root、同一 S0A 合同或在原合同内修 parser 后重跑均未获授权。

## 已可靠建立的事实

- 执行前 `HEAD == origin/master == b65c0d9...`，formal Git/hash gate 通过；
- attempt 与 preflight 在首个 Git 网络请求前 durable；
- formal CLI 只调用一次；
- canonical failure 被冻结的 terminal validator 接受；
- failure 明确记录没有数据托管端请求、没有打开数据 ZIP；
- `catalog.json` 与 `result.json` 均不存在，因此没有目录父体数或环境数。

终态后仅做本地控制面核对：`toolkit/FETCH_HEAD` 指向冻结提交
`158a6844d782942110967325ca3082f50ab2bfc7`，`.git/modules` 与工作树
`.gitmodules` 均不存在；没有重新打开第 978 行或任何 manifest 内容。该核对不在
canonical failure 哈希链中，不参与目录容量或 source claim。

收据 SHA-256：

- attempt:
  `4a5b65a2a53ecfb343c50bff4929f03e8c0f109695df509098d3b2d499cf3ac8`
- preflight:
  `0a5b9514e9a7332249c44169757551f051f79d128fe5cd4a392abbe4c6ed9652`
- failure:
  `28f4c0337935a0778d1a9ea58c89de559779d85d59d919347a948140d6dd7fd5`

## 后继边界

若继续，只能先冻结一个新版本控制面协议，使用新 canonical root，并在执行前完成
独立审计；它不能被描述为原 S0A 的 resume/retry，也不能自动获得执行权。S0B、
数据 payload、ecology、effect、student、主线/App/Android、生产与 safety 仍全部关闭。

机器结果 SHA-256：
`f86153427117ed8542cb892204a693805b80b0f4eac87cdf18c26e9d2aad4961`。
[机器结果 JSON](HFTF_STAGE_C_D5_S0A_TARTANGROUND_CATALOG_INVALID_RESULT_2026-08-02.json)。
