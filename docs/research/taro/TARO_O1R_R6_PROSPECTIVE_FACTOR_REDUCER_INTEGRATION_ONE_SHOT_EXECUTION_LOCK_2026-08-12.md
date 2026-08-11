# TARO O1R R6 reducer integration one-shot execution lock

状态：`FROZEN / ONE_SHOT / OUTPUT_ROOT_MUST_BE_ABSENT / SOURCE_ONLY`

机器合同：[JSON](TARO_O1R_R6_PROSPECTIVE_FACTOR_REDUCER_INTEGRATION_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json)

本锁授权且只授权在 16 个冻结 eval parent、239 帧、2151 query 上重放已经冻结的 R6 factor bundle 与 O1R reducer。运行时只可从 source archive 打开 `confidence`；candidate、R6 bundle 与 fit-only uncertainty artifact 都由 hash-bound 本地证据提供。`highres_depth`、`lowres_depth`、`color`、训练、网络、阈值调整、App/设备修改均为零。

输出根 `artifacts.local/evidence/taro/o1r-r6-reducer-integration-r0` 必须不存在，创建根即消费 one-shot；禁止覆盖和重跑。无论结果如何都不授权 promotion 或产品/安全结论。

唯一后继：`TARO_O1R_R7_POSITIVE_OCCUPANCY_AND_CLEAR_COVERAGE_TASK_LOCK`。
