# TARO O1R R6 factor → reducer implementation lock

状态：`FROZEN / 13_OF_13_FOCUSED_TESTS_PASS / EXECUTION_NOT_AUTHORIZED_BY_THIS_LOCK`

机器合同：[JSON](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_IMPLEMENTATION_LOCK_2026-08-12.json)

实现已经把 sealed R6 factor bundle、source-only query uncertainty lookup、R3 fit-only uncertainty model 与唯一 interval reducer 连成一条可执行链。`QUERY_CLEARANCE` 只能来自 `R1_BASELINE`；lookup、uncertainty 和 query result 分层独立 seal；结构/hash 漂移整包拒绝，普通缺失证据逐 query 保留 `UNKNOWN`。

真实封存 uncertainty artifact 已成功 hydrate 为 factory-bound model，模型 SHA-256 为 `3FB93A...5365`。聚焦测试覆盖公开 API firewall、9-query cardinality、real model hydrate、confidence/K/owner mutation、missing query、nested uncertainty mutation、determinism 和 uncertainty monotonicity，共 13/13 通过。

本锁不授权数据执行。唯一后继：`TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_ONE_SHOT_EXECUTION_LOCK`。
