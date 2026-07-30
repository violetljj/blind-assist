# Dual-loop causal radial geometry LITE R0

状态：implementation review pass；full replay 尚未激活

## 研究问题与版本

本 Module 服务于 `DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0`：
在 source-GT target/ROI 条件下，比较 causal box log-area growth 与 ROI sparse
radial flow，判断它们是否值得进入新的独立 Confirmation。Development 输入、
truth-only natural-event ledger、两臂 producer、post-hash evaluator 与 synthetic
fixtures 已实现并通过独立实现检查；全量 replay 尚未激活。

## 稳定 Interface

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/dual_loop_radial_geometry_lite_r0/prepare_revel_development_manifest.py `
  --bag-root artifacts.local/evidence/datasets/revel-dynamic-bag-v1-20260720 `
  --image-label-root artifacts.local/evidence/datasets/revel-dynamic-images-labels-v1-20260720 `
  --output-root artifacts.local/evidence/dual-loop/target-track-causal-radial-geometry-lite-r0/input-freeze
```

脚本只读 source RGB/label/Vicon，生成：

- `replay_input.jsonl`：候选进程唯一可见的 RGB/ROI/opaque track allowlist；
- `truth.jsonl`：只允许 evaluator 读取的 Vicon truth 与 natural-event join；
- `natural_events.jsonl`：truth-only parent event；
- `manifest.json`：输入/输出 SHA-256、固定分母和嵌套层级。

## 输出

只写入传入的 `artifacts.local/` 目录。所有 JSON 使用 UTF-8 和确定性排序。

实现入口为 `run_replay.py` 和 `evaluate_replay.py`。两者只能按已评审的一次性
activation decision 执行；本 README 不构成执行权限。

## 安全边界

- REveL 只承担 Development truth；Vicon 不得进入候选 producer。
- `track-000/001` 是 opaque replay ID；green/yellow 映射只存在于 evaluator truth。
- REveL 是单一 capture；target、event、frame 都不伪装成跨 capture 独立样本。
- 不读取旧 F-1B decision 输出，不产生 Android、提醒、产品或安全 authority。

## 停止条件

输入哈希、时间单调性、RGB/label/bag 一一对应、target 唯一性或 truth ancestry
任一不闭合即停止本 evidence version。若无法形成两目标、三 anchor-region 的
truth-only parent events，则设计保持 `HOLD`，不得实现候选 replay。

## 假设与规则质疑

完整 8,580-frame capture 是解决稀疏 512-frame Discovery 账本不能形成因果窗口和
自然事件分母的最小修复。它仍然只有一个 capture，因此只支持 Development 描述，
不支持 confirmation 推断。

## 失败资产复用

失败的 manifest 可保留为 source-integrity diagnostic；不得改名为 confirmation
evidence，也不得用候选输出来选择替代窗口。
