# RCLE Synthetic Scene Mechanism Validation R0 development result

终态：`SYNTHETIC_MECHANISM_DEVELOPMENT_PASS`

权限上限：
`CONTROLLED_SYNTHETIC_CAUSAL_MECHANISM_EVIDENCE_ONLY`。

## 执行与验证

- split：`development`；2 个 scene family × 6 种 motion，共 12 个序列。
- 每序列 101 帧、100 pair；总计 1212 帧、1200 pair。
- dataset receipt：
  `7a09cac32dd6b4e43d70e6b4b88c3b6d2d70988f53112d73d4d205e6288f4097`。
- 独立 dataset validator：`PASS`；完整 frame/scene/algorithm manifest、payload
  hash、timestamp、signed K、pose inverse、depth 与 split 均通过。它只对 algorithm
  ledger 做 JSONL parse check，不声称独立重算 scientific outcome；科学门由冻结
  evaluator 生成并另存 ledger hashes。
- 第二次 fresh development 生成与第一次的三个 manifest 和全部有序 payload
  hash 相同；按冻结规则只排除 `generated_at` 与 `environment.pid` 后 receipt
  相同，determinism 为 `PASS`。
- 单元与错误注入测试：25/25 `PASS`；staged RGB 或 valid mask 改一个字节均在
  algorithm 执行前 fail closed；depth、full-pose translation、role 和 surface-id
  四类 truth-only mutation 均不改变独立重放的 algorithm ledger。
- RGB 时间轴视觉 QA：每个序列每 5 帧取一帧，共检查 252/1212 帧
  （20.79%）。未见缺帧、黑帧、纹理跳变或与 motion role 不一致的路径。

## 冻结门结果

12/12 个 sequence-local gates 通过，未使用 pooled rescue。

| scene | motion | old trigger | R1 trigger | R1 coverage | first delay |
|---|---:|---:|---:|---:|---:|
| corridor_grid_dev | positive_approach | 83 | 75 | 0.75 | 0.20 s |
| corridor_grid_dev | positive_rotation_approach | 99 | 95 | 0.95 | 0.20 s |
| corridor_grid_dev | below_static | 0 | 0 | 0.00 | n/a |
| corridor_grid_dev | below_pure_yaw | 11 | 0 | 0.00 | n/a |
| room_panels_dev | positive_approach | 100 | 98 | 0.98 | 0.20 s |
| room_panels_dev | positive_rotation_approach | 100 | 98 | 0.98 | 0.20 s |
| room_panels_dev | below_static | 0 | 0 | 0.00 | n/a |
| room_panels_dev | below_pure_yaw | 10 | 0 | 0.00 | n/a |

两个 lateral 与两个 receding stress 序列按预注册只报告、不参与 PASS；四个序列
R1 trigger 均为 0。positive 的 algorithm median expansion 为
0.0789–0.0938/s；pure-yaw truth 约为 0，补偿后 algorithm median 为
-0.00323 至 -0.00244/s。rotation+approach 与 approach 的同场景差分别为
0.01418/s 和 0.00336/s，均通过冻结门。

ledger hashes：

- truth：
  `086c0b27e66e627b7e0d57c8fa23cb1e83dfbd600391308115238d9aa00e79e6`
- algorithm：
  `f1e1aae6635c3a3bdaa50c267b58d5dc9535ca34531cf0dfb55811cf0059e021`
- old/R1 comparison：
  `17e6ece8835561cbb6271c167655ed0e7bcfac98e62dd1d7b7fe6213b565d6f9`

## 解释边界与下一步

本结果支持“冻结 RCLE/R1 机制在这两个受控 synthetic development scene family
中按预期区分接近与低参考运动”。它不能替代真实数据 external confirmation，
不能回写既有 `NOT_EVALUABLE`，也不能授权 Android、产品或安全晋级。

后续 sealed split 已在独立治理核对、hash-bound activation 和独立授权后消费唯一
claim。它的不可重跑终态为 `SYNTHETIC_MECHANISM_SEALED_REVISE`，详见同日 sealed
结果文档；本 development PASS 不得用于回救 sealed failure。
