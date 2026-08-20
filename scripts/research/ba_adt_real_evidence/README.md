# BA-ADT Real Evidence

状态：`current / REVERSIBLE_EXPLORATION / ADT-0-SAMPLE-MINED-PARTIAL-EVENT-COVERAGE / FULL-SEQUENCE-SELECTION-NEXT / SKY-DISABLED / DEFAULT-APP-UNCHANGED`

## 目标与边界

`BA-ADT-REAL-EVIDENCE` 用真实 Aria 第一视角 RGB 建立 Goal Copilot 的视觉证据链。ADT ground truth
只能用于 episode mining 和对 RGB estimator 的旁路评价，绝不能作为 detector、tracker、bearing、
nearness、approach 或 observation-quality estimator 的输入。

官方来源：[ADT overview](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset)、
[download guide](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download)、
[data format](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/data_format)。

ADT 是 prerecorded trajectory。本路线可回答真实 RGB 是否支持稳定的 target visibility、bearing、
tracking、reacquisition、relative nearness 和 approach evidence；不能证明系统引导改变了用户动作，
也不能声称 closed-loop navigation、真实用户安全或默认 App 可用性。

路线顺序固定为：

1. `ADT-0`：GT-only episode mining，判断数据是否自然包含 Goal Copilot 事件；
2. `ADT-1`：RGB-only estimator，GT 仅在隔离 evaluator 中计分；
3. `ADT-2`：接冻结 Goal Copilot，输出 prerecorded guidance timeline；
4. `ADT-3`：只有 RGB failure 明确归因到 policy 层时，才允许另立 Sky task。

当前只激活 `ADT-0`。不得启动 Sky、GC2-C、held-out、Android/default-App 接线或导航结论。

## 稳定 Interface

下载官方 sample 的 RGB preview 与 main ground truth（当前 manifest 合计必须小于 32 MiB）：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence acquire_sample.py `
  --output-dir artifacts.local/datasets/ba_adt_real_evidence/sample `
  --receipt artifacts.local/evidence/ba_adt_real_evidence/sample/acquisition.json
```

GT-only episode mining：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence mine_goal_episodes.py `
  --groundtruth artifacts.local/datasets/ba_adt_real_evidence/sample/ADT_Apartment_release_golden_skeleton_seq100_10s_sample_M1292_main_groundtruth.zip `
  --output artifacts.local/evidence/ba_adt_real_evidence/sample/episodes.json
```

## 输出

机器输出位于 ignored `artifacts.local/`。源码只记录来源、身份、阈值、事件覆盖与 claim ceiling。

## 安全边界

- RGB 与 GT 使用不同角色字段和不同文件；GT 不得进入 estimator。
- visibility gap 只是 mining candidate，未经复核不得称为遮挡或 tracker failure。
- prerecorded replay 不产生 interactive navigation、用户安全、产品或默认 App 权限。
- 当前不运行 Sky、模型搜索、GC2 held-out 或 Android 接线。

## 停止条件

若官方 manifest 身份漂移、sample 超过 32 MiB、下载 hash 不符、GT schema 缺失，立即 fail closed。
若少量完整 sequence 仍不能产生适合的自然多阶段 episode，记录 ADT 数据适配性边界并评估其他真实
第一视角来源；不得降低门槛或把多个无关目标拼成一个完整 episode。

## 当前 successor

Sample 已得到 102 个持续跟踪候选，覆盖全部六类事件，但没有单一目标覆盖完整六阶段；详见
[`BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md`](../../../docs/research/goal-copilot/BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md)。
下一步通过 Dataset Explorer 选择少量完整 sequence；不能为了凑齐六阶段而事后改写事件定义。只有
ADT-0 找到合适的自然 target episode 后才实现 ADT-1 RGB adapter。
