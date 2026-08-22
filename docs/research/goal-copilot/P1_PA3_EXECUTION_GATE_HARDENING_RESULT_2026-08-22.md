# P1 PA3 execution-gate hardening result

状态：`IMPLEMENTED / DENOMINATOR_INSUFFICIENT / PROVIDER_CALLS=0 / PA3_INFERENCE_NOT_AUTHORIZED`

## 结论

PA3 的文档门现在变成了真实可执行门。审计前，`materialize_pa3_inputs.py` 没有验证 prospective physical capture
manifest 的自有 body hash/device provenance；`run_yoloe_semantic_prompt.py` 也能在没有 visibility-denominator receipt
时直接启动。两条旁路均已 fail closed。

当前链条为：

```text
C0 public Goal Contract
→ hash-bound device capture plan / receipt / fixed-offset frames
→ private truth
→ PA3 public/private materialization
→ private denominator authorization (>=5 visible episodes AND >=8 visible frames)
→ one authorized semantic-only prediction path + dispatch journal
→ YOLOE semantic proposals
→ completed-journal-bound private evaluation
```

## 新强制条件

- physical capture manifest 必须验证 body hash、C0 hash、device receipt hash、capture plan hash、source role、全局
  capture instruction、设备视频时间语义、固定结束前 `2.5/1.5/0.5 s` 抽帧与 `provider_model_calls=0`；
- `authorize_pa3.py` 从绑定的 private truth 只导出 aggregate denominator，不向 provider 暴露 case identity/truth；
- 少于 `5` 个 visible episodes 或 `8` 个 visible frames 时 receipt 固定为
  `NOT_EVALUABLE_INPUT_CONTRACT / pa3_inference_authorized=false`；
- 授权 receipt 只允许 `YOLOE_26N_SEG_GOAL_SEMANTIC_TEXT_PROMPT_ONLY`，明确禁止 FRG、identity 与任何 sweep；
- receipt 绑定唯一 prediction 与 dispatch journal 路径；runner 在读取模型/导入 YOLOE 前验证授权；
- 每个 `model.predict` 在调用前 journal 为 dispatched，成功后才计 completed。失败、中断或已有 journal/output
  均禁止 retry/replay；
- evaluator 必须验证同一 authorization、`COMPLETED` journal、prediction hash 与 dispatched/completed call accounting。

## 当前验证与边界

- proposal-availability focused tests：`43` 项通过；其中 prompt-embedding failure 会封存为 `in_doubt=1`；
- Python compile check：通过；
- 低 denominator 测试证明 runner 在访问不存在的 model/prompt 文件之前返回 `not authorized`，且不创建 journal；
- Android environment health：SDK/ADB 可用，ready devices `0`，AVDs `0`；
- 本轮 provider/model calls：`0`。

因此当前仍不是 PA3 负结果，也不是算法效果结果。下一动作保持为真实设备完成 frozen prospective cohort，随后私有标 truth。
只有新授权 receipt 为 true，才执行一次固定 YOLOE semantic proposal availability；成功再授权 contrastive verifier，失败才进入
functional/region grounding。
