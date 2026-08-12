# DepthART-S D2 Development quality terminal

## 结论

冻结的 D2 Development screen 终态为
`D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_STOP`。

设备阶段完成 24/24 chunks、4 个 identity、1,200/1,200 帧和 10,800 cells。1,200 个
saved-context 输出共 1,307,443,200 bytes，独立 validator 已逐文件重算 SHA，并确定性复现质量结果。

该结果不表示 head 没有作用。相对 same-base no-head baseline，冻结 head 将 pooled clearance MAE
从 `0.436153 m` 降至 `0.279311 m`，false-clear 从 `0.207949` 降至 `0.085991`，temporal
clearance delta MAE 从 `0.112629 m` 降至 `0.094103 m`，geometry transition agreement 从
`0.729193` 提高至 `0.850526`。但是预注册合同要求同时满足全部绝对门、noninferiority 门和 finite
strata，因此整体必须判 FAIL。

## 失败门

绝对门失败：

- clearance MAE：`0.279311 m > 0.20 m`；
- false-clear：`0.085991 > 0.08`；
- false-block：`0.376138 > 0.02`；
- geometry transition agreement：`0.850526 < 0.90`；
- worst-parent false-clear：`0.159503 > 0.12`。

相对 baseline 的 noninferiority 失败：known coverage、false-block、valid-to-unknown。冻结 head 的
known coverage 从 `0.991474` 降到 `0.964858`，false-block 从 `0.262735` 升到 `0.376138`，
valid-to-unknown 从 `0.008526` 升到 `0.035142`。

此外，baseline 与 candidate 的 `center@2.0m`、`left@2.0m` false-block denominator 均为空；
按冻结协议的 missing-denominator 规则，required strata fail-closed。

## 暂停与恢复

用户暂停时已完成 17 个 chunks。正在运行的 chunk 17 被明确终止，退出码 143 的部分设备输出没有
生成 receipt、没有被消费；远端临时目录已清理。恢复后使用原 materialization receipt 从头执行
chunk 17 的完整设备阶段，随后顺序完成 chunks 18–23。

## 权限边界

D2 Development outcome 已消费，不得通过修改 head、阈值、postprocess、数据、denominator 或 gate
回救本次结果。R2 candidate 不授权，R2 cohort 继续 sealed；性能、DA2 替换、默认 App、production
和 safety 均未授权。

当前没有自动 successor。任何新假设必须建立新的版本、pre-outcome 合同和 fresh data，并重新取得
显式授权。

机器证据见 [governed result](DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_QUALITY_RESULT_2026-08-12.json)。
