# CID-SIMS floor3_1 disjoint holdout R0

状态：development / frozen

## Stable Interface

`formal_runner.py` 是唯一正式入口。它先只读取 ZIP central-directory、绑定的
`groundtruth.txt`、depth PNG 和时间戳，生成 geometry ledger 与冻结选窗结果。
只有精确的 `2 positive + 2 below-reference` 窗满足 guard、coverage、持续性与
20 秒起点间隔后，才独占写入 RGB identity lock、读取所选 color PNG，并逐窗调用
旧 development canary 的未修改 `evaluate_window()`。

## 输出

只写
`artifacts.local/evidence/rcle_rgb_algorithm_cid_sims_floor3_1_disjoint_holdout_r0/`。
大数据、cache、ledger、claim、progress、validation 与 terminal receipt 均不入库。

## 安全边界

本轮只能回答同一 `floor3_1` sequence 的未查看互斥窗口是否复现方向区分；不是
cross-source confirmation、性能资格、独立算法实现确认或产品/安全证据。候选角色
不足时必须 `NOT_EVALUABLE`，禁止读取 RGB member bytes、换窗、滑窗、降门或补位。

## 停止条件

contract、implementation lock、archive、算法传递绑定、runtime、window identity、
guard、spacing、ledger、aggregate 或 authority 漂移均判 INVALID。geometry 角色
不足或 RGB coverage 不足为 `NOT_EVALUABLE / VALID`；abstention 不填补。

