# BA-ADT Real Evidence

状态：`current / REVERSIBLE_EXPLORATION / P1-D0-TEMPORAL-DEVELOPMENT-COHORT-READY / R5-PERMANENTLY-CLOSED / NO-DINOV2-SKY / DEFAULT-APP-UNCHANGED`

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

当前已执行 ADT-0、ADT-1 Development evaluation 与首个 ADT-2 prerecorded demo。不得启动 Sky、
GC2-C、held-out、Android/default-App 接线或导航结论。

ADT-1 的 RGB-only mechanical canary 可使用 `run_rgb_observer.py`，再由独立
`evaluate_rgb_observations.py` 读取 prediction + GT。Observer CLI 没有 GT 参数；bearing 明确是
normalized image-x，nearness 是 bbox-scale proxy，二者都不能冒充标定角度或 metric range。

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

P1-D0 temporal cohort 只读取 ADT source identity、timestamp、visibility 与 bbox truth，并用 MP4 自带的
逐帧 timestamp metadata 对齐 RGB；选择过程不读取任何 tracker/model output：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence `
  materialize_p1_temporal_cohort.py `
  --source <SEQUENCE_ID> <GROUNDTRUTH_ZIP> <PREVIEW_RGB_MP4> `
  --output-dir artifacts.local/evidence/p1_d0_temporal_cohort_v1 `
  --episode-budget 15
```

固定合同与 Development 终态见
[`P1-D0 protocol`](../../../docs/research/goal-copilot/P1_D0_TEMPORAL_COHORT_PROTOCOL_V1.md)。

Sample 之后，对显式选择且总量有界的完整 sequence 只下载 main GT：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence acquire_sequence_groundtruth.py `
  --sequence-id <ADT_SEQUENCE_ID> `
  --output-dir artifacts.local/datasets/ba_adt_real_evidence/selected_gt `
  --receipt artifacts.local/evidence/ba_adt_real_evidence/selected_gt/acquisition.json
```

只有 GT mining 选中 episode 后，才单独下载对应 preview RGB：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence acquire_sequence_rgb.py `
  --sequence-id <ADT_SEQUENCE_ID> `
  --output-dir artifacts.local/datasets/ba_adt_real_evidence/selected_rgb `
  --receipt artifacts.local/evidence/ba_adt_real_evidence/selected_rgb/acquisition.json
```

RGB-only observation、隔离评价与 demo 渲染：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence run_rgb_observer.py `
  --video <PREVIEW_RGB_MP4> --model <YOLO_MODEL> --target-class carrot --flow-max-gap 5 `
  --instance-redetection --device 0 `
  --output <OBSERVATIONS_JSON>

E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence evaluate_rgb_observations.py `
  --observations <OBSERVATIONS_JSON> --groundtruth <GROUNDTRUTH_ZIP> `
  --target-uid 4917588638317799 --output <EVALUATION_JSON>

E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence account_redetection_failures.py `
  --observations <CANDIDATE_DIAGNOSTIC_OBSERVATIONS_JSON> --groundtruth <GROUNDTRUTH_ZIP> `
  --target-uid 4917588638317799 --output <FAILURE_ACCOUNTING_JSON>

E:\codex-tools\bin\blindassist-python.cmd scripts/run_research_tool.py ba-adt-real-evidence build_offline_demo.py `
  --video <PREVIEW_RGB_MP4> --observations <OBSERVATIONS_JSON> --evaluation <EVALUATION_JSON> `
  --groundtruth <GROUNDTRUTH_ZIP> --target-uid 4917588638317799 --target-name Carrot_A `
  --policy scripts/research/goal_copilot_2a/frozen_gc1_winner.py `
  --output-video <DEMO_MP4> --output-timeline <TIMELINE_JSON> --contact-sheet <CONTACT_SHEET_PNG>
```

Failure accounting 要求 observer 同一次重放显式增加 `--candidate-diagnostics`；该开关只记录 RGB-only
proposal/verifier trace，不向 observer 暴露 GT。YOLOE 单变量 arm 还需增加
`--redetection-generator yoloe-visual-prompt --redetection-model <YOLOE_CHECKPOINT>`，正常 YOLO detector、
flow、TargetMemory 和 confirmation 参数保持不变。

固定五窗口 observability audit 使用 `audit_reappearance_windows.py` 生成 GT-derived oracle proposal
maps、逐窗 JSON 与 contact sheet。oracle arm 必须使用
`--redetection-generator diagnostic-oracle --diagnostic-oracle-proposals <WINDOW_MAP>`；它会把 output 标成
非 RGB-only，现有正式 evaluator 必须拒绝。五窗必须分别独立运行，不能让前窗注入改变后窗状态。

R4 只改变 LOST search。S1 使用
`--redetection-generator highres-full-frame --redetect-imgsz 1280`；S2 使用
`--redetection-generator tiled-full-frame --redetect-imgsz 640 --redetect-tile-overlap 0.20`。后者固定为
2x2 tiles 并把局部 bbox 映射回全图后去重。`evaluate_search_scale_r4.py` 用 `--arm NAME=PATH` 接收
S0/S1/S2，固定读取 R1 五窗；它显式记录 `T_invisible/T_subdetectable/T_system_miss/T_confirmation`，
proposal 未出现时将 `T_system_miss` 在窗口末尾标为右删失。

R5 Attempt 01 的 SAM 3 入口 `run_visual_upper_bound_r5.py` 在 inference 前因 8 GB 本机机械不可行而
supersede。正式 Attempt 02 的 GT-blind OWLv2-large 入口是 `run_visual_upper_bound_r5_owlv2.py`，隔离评价
入口仍是 `evaluate_visual_upper_bound_r5.py`。完整冻结参数、输入 SHA、合法 exemplar 选择和三分支终止门见
[`R5 Attempt 02 protocol`](../../../docs/research/goal-copilot/BA_ADT_SMALL_TARGET_VISUAL_UPPER_BOUND_R5_ATTEMPT_02_PROTOCOL_2026-08-21.md)。

## 输出

机器输出位于 ignored `artifacts.local/`。源码只记录来源、身份、阈值、事件覆盖与 claim ceiling。

## 安全边界

- RGB 与 GT 使用不同角色字段和不同文件；GT 不得进入 estimator。
- visibility gap 只是 mining candidate，未经复核不得称为遮挡或 tracker failure。
- prerecorded replay 不产生 interactive navigation、用户安全、产品或默认 App 权限。
- 当前不运行 Sky、模型搜索、GC2 held-out 或 Android 接线。

## 停止条件

若官方 manifest 身份漂移、sample 超过 32 MiB、下载 hash 不符、GT schema 缺失，立即 fail closed。
当 manifest transport 不稳定时，可先把同一官方 JSON 缓存到 `artifacts.local/`，再用
`--manifest-file` 读取；receipt 会绑定缓存 SHA-256，不能手工改写 URL 或成员身份。
若少量完整 sequence 仍不能产生适合的自然多阶段 episode，记录 ADT 数据适配性边界并评估其他真实
第一视角来源；不得降低门槛或把多个无关目标拼成一个完整 episode。

## 当前 successor

P1-D0 已从 2 条现有 ADT GT/RGB source 自动物化 15 episodes / 15 physical targets / 1,724 frames，
六类 temporal mode 均非零，终态 `P1_TEMPORAL_DEVELOPMENT_COHORT_READY`。该批只服务 P1 Development；
EgoTracks fallback 不触发，不再追加数据。后续只允许 P1 representation/evaluator/baseline 消费该 cohort，
不得把 ADT indoor object 结果称为 entrance persistence、fresh confirmation 或产品证据。

Sample 已得到 102 个持续跟踪候选，覆盖全部六类事件，但没有单一目标覆盖完整六阶段；详见
[`BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md`](../../../docs/research/goal-copilot/BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md)。
固定门槛已在 `clean_seq134/136` 找到 172/134 个六阶段候选，首选 `seq136 / Carrot_A`；详见
[`BA_ADT_REAL_EVIDENCE_ADT0_SELECTION_ADT1_CANARY_RESULT.md`](../../../docs/research/goal-copilot/BA_ADT_REAL_EVIDENCE_ADT0_SELECTION_ADT1_CANARY_RESULT.md)。
修正 90° clockwise 坐标变换后，sample `bowl` canary 仍定位到多实例 grounding failure。完整
`seq136 / carrot` held-forward evaluation 显示定位成功时 bearing/scale/approach 有信号。5-frame sparse
optical-flow candidate 将 recall 从 0.4041 提至 0.5808，GT-invisible false-visible 为 0.0073，已准入
ADT-2 Development demo。实例重检测 R1 随后以多模板 appearance memory、弱时空先验和 2-of-3
确认把 recall/mean IoU 提至 `0.6203/0.4743`，13 次重检测为 `13/0/0 correct/wrong/unresolved`，且
false-visible 不变；但 @30/@90/@180 reacquisition 仍为 `0.4/0.5/0.5`，最长 dropout 只降到 159 帧。
结果见
[`BA_ADT_INSTANCE_REDETECTION_1_RESULT_2026-08-21.md`](../../../docs/research/goal-copilot/BA_ADT_INSTANCE_REDETECTION_1_RESULT_2026-08-21.md)。
后置 accounting 显示 5 个失败全部为 `NO_CANDIDATE`，R1 candidate recall during LOST 仅
`34/405 = 0.0840`。只替换候选生成器的 YOLOE-26n visual-prompt canary 未改善任何失败窗口，candidate
recall 降至 `29/423 = 0.0686`、@30 降至 `0.2`、最长 dropout 增至 164，wrong-instance 仍为 0；结果见
[`BA_ADT_YOLOE_VISUAL_PROMPT_CANARY_RESULT_2026-08-21.md`](../../../docs/research/goal-copilot/BA_ADT_YOLOE_VISUAL_PROMPT_CANARY_RESULT_2026-08-21.md)。
R3 五窗口审计得到 oracle `4/5`；唯一失败只有 1 帧 3×3 px，2-of-3 不可能完成。五窗分为 2 个
不可见/重遮挡和 3 个太小；640 observer 输入上的目标最短边最大值均小于 10 px，没有建立尺度足够的
model miss 或 identity ambiguity。结果见
[`BA_ADT_REAPPEARANCE_OBSERVABILITY_R3_RESULT_2026-08-21.md`](../../../docs/research/goal-copilot/BA_ADT_REAPPEARANCE_OBSERVABILITY_R3_RESULT_2026-08-21.md)。
R4 的 S0/S1/S2 在 3 个 eligible 窗口、97 个 LOST-search frames 上均为 `0/97` candidate recall、
`0/3` confirmed reacquisition；S1/S2 全局 recall 改善却分别产生 1/4 次 wrong-instance，故不保留。
结果见
[`BA_ADT_SMALL_TARGET_SEARCH_SCALE_R4_RESULT_2026-08-21.md`](../../../docs/research/goal-copilot/BA_ADT_SMALL_TARGET_SEARCH_SCALE_R4_RESULT_2026-08-21.md)。
R5 Attempt 01 的 SAM 3 在 teacher outcome 前被机械可行性否决，未消费固定窗口。Attempt 02 已冻结
OWLv2-large Teacher A、合法 RGB exemplar、3-window/97-frame denominator 与 `0/3, 1/3, >=2/3` 终止门。
唯一下一动作是运行该固定 teacher/evaluator；不继续扫 R4，不接 DINOv2/Sky。
