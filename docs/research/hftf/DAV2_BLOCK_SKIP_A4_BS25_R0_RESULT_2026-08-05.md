# DA V2 A4-BS25 固定 block-skip 结果

日期：2026-08-05

终点：`MODEL_VARIANT_R2_ENGINEERING_NONINFERIORITY_FAIL`

候选严格使用冻结的 518/FP16 checkpoint，只跳过零基索引 `3/7/11` 三个 block。唯一缓存
SHA-256 为 `6854CDA8...FD81`。第一次 R1 preflight 因 `clearance_mae_m=null` 触发 evaluator
类型错误且未生成结果；随后冻结的 R2 只新增“未定义指标必失败”，canonical self-check
仍为 14/14，再对同一缓存完成判定。

## 结果

- 仅通过 `3/14` 门，失败 11 个；
- raw AbsRel `166.22%`，scale-aligned AbsRel `14.36%`；
- ground recovery `72.27%`，paired-valid frames `71.67%`；
- 86 帧可形成真值几何 state pair，但 0 个 collision decision 可比较；
- clearance、collision、false-clear、false-block、temporal 与 harmful-change rate 均不可定义，
  按 R2 明确失败，绝不解释成 false-clear 为零。

主机 CUDA cache 物化 P95 `48.42 ms` 只表明跳层确实减少了计算；它不是 Android QNN/App
性能，也不能补偿灾难性质量退化。因此 A4-BS25 停止，不进行导出、转换或设备 profile。

完整 P1-R2 结果 SHA-256：`9622E2CA...B9E2`。
