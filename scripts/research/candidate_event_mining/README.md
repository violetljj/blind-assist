# candidate_event_mining

状态：`discovery / THESIS_DEVELOPMENT / candidate-pool-only`

本 Module 为长视频和公开数据建立模型无关的候选事件挖掘流水线。它只回答“哪些窗口值得独立复核”，不回答事件真值、算法效果、安全性或产品可用性。当前版本冻结为 `CANDIDATE_EVENT_MINING_DISCOVERY_R0`。

## 研究问题与版本

问题是能否用批量推理产生的逐帧响应，在不读取既有确认性结果的前提下，稳定地找到以下 review 候选：正前方障碍接近、横穿、用户接近静态障碍、台阶/落差、平行路沿、门框/桌角/树枝、正常通行负例、转头/抖动负例、动态人群、YOLO 低覆盖但 segmentation/depth 有响应、segmentation 高频响应和 HFTF future-field 明显变化。

候选触发是模型/规则响应，不是标签。`segmentation_high_frequency_alert` 也不等于“误提醒”；它只表示值得让 Luna 检查是否为高频误触发。`yolo_miss_segmentation_or_depth_response` 只表示 detector coverage 与另一通道响应不一致，不产生漏检率结论。HFTF 字段只作为 discovery sidecar，不改变 HFTF 当前自己的 stage、source 或 execution contract。

Forward governance 固定为 `THESIS_DEVELOPMENT`。本 Module 不打开现有 dual-loop/HFTF confirmation、fresh holdout、Android、默认 App 或生产权限；候选池也不能直接成为训练集或事件真值。需要 claim-critical 事件结论时，必须另立新的 protocol/data-role/review contract。

## 稳定 Interface

从仓库根目录使用稳定 Adapter：

```powershell
$py = 'E:\codex-tools\bin\blindassist-python.cmd'
$tool = 'scripts/run_research_tool.py'
$contract = 'configs/candidate_event_mining_contract_v1.json'
$index = 'F:\ba-data\blindassist-candidate-event-mining\project_index.json'

& $py $tool candidate-event-mining init_project_index.py `
  --output $index

& $py $tool candidate-event-mining ingest_batch_inference.py `
  --contract $contract `
  --project-index $index `
  --input <adapter-output-a.jsonl> `
  --input <adapter-output-b.jsonl> `
  --run-id <batch-run-id> `
  --adapter-id <adapter-name-and-version> `
  --output artifacts.local/evidence/candidate-event-mining/<run-id>/batch_trace.jsonl

& $py $tool candidate-event-mining mine_candidates.py `
  --contract $contract `
  --project-index $index `
  --input-trace artifacts.local/evidence/candidate-event-mining/<run-id>/batch_trace.jsonl `
  --run-id <run-id> `
  --output artifacts.local/evidence/candidate-event-mining/<run-id>/candidate_report.json

& $py $tool candidate-event-mining build_review_bundle.py `
  --contract $contract `
  --candidate-report artifacts.local/evidence/candidate-event-mining/<run-id>/candidate_report.json `
  --output artifacts.local/evidence/candidate-event-mining/<run-id>/review-bundle

& $py $tool candidate-event-mining finalize_luna_reviews.py `
  --contract $contract `
  --candidate-report artifacts.local/evidence/candidate-event-mining/<run-id>/candidate_report.json `
  --review-bundle artifacts.local/evidence/candidate-event-mining/<run-id>/review-bundle `
  --reviews artifacts.local/evidence/candidate-event-mining/<run-id>/luna_reviews.jsonl `
  --output artifacts.local/evidence/candidate-event-mining/<run-id>/candidate_pool.json
```

当全量报告大于一次独立视觉 review 的合理预算时，先保留完整 `candidate_report.json`，再生成一个有父报告 hash 的 review queue；未进入 queue 的候选不是 reject，也不进入本次 pool：

```powershell
& $py $tool candidate-event-mining select_review_queue.py `
  --contract $contract `
  --candidate-report artifacts.local/evidence/candidate-event-mining/<run-id>/candidate_report.json `
  --max-candidates 64 `
  --output artifacts.local/evidence/candidate-event-mining/<run-id>/review_queue_report.json
```

随后将 `build_review_bundle.py` 和 `finalize_luna_reviews.py` 的 `candidate-report` 指向 `review_queue_report.json`；全量报告仍是完整发现库存，queue 的 `review_queue.unreviewed_candidate_count` 保留未复核分母。

### 真实公开视频 batch adapter

`run_real_video_batch.py` 是本 Module 的一个 bounded host adapter，不是候选核心的替代实现。它从已登记的 `F:\ba-data` media path 顺序解码，按不超过约 500ms 的 cadence 抽取 JPEG review frames，实际运行 YOLO11n 与 Depth Anything V2，并写出 canonical frame JSONL 与独立 adapter manifest：

```powershell
$py = 'E:\codex-tools\bin\blindassist-python.cmd'
$tool = 'scripts/run_research_tool.py'

& $py $tool candidate-event-mining run_real_video_batch.py `
  --project-index F:\ba-data\blindassist-candidate-event-mining\project_index.json `
  --run-id <run-id> `
  --output artifacts.local/evidence/candidate-event-mining/<run-id>/adapter_trace.jsonl `
  --sample-fps 2 `
  --device cuda `
  --yolo-model artifacts.local/models/yolo11n.pt `
  --depth-checkpoint artifacts.local/models/dg-srf-f0/depth_anything_v2_vits.pth `
  --depth-source-root artifacts.local/models/dg-srf-f0/source-a561b849/Depth-Anything-V2-a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
```

`--enable-segmentation-proxy` 只启用一个低级图像空间风险 proxy，manifest 会明确记录 `image_space_risk_proxy_not_a_segmentation_model`；不启用时 segmentation 信号保持缺失。`--segmentation-sidecar` 与 `--hftf-sidecar` 必须按 `source_id/session_id/frame_index` 精确绑定，缺失的通道不补零、不生成负证据。完整 chain 仍由 `ingest_batch_inference.py`、`mine_candidates.py`、`build_review_bundle.py` 和 `finalize_luna_reviews.py` 依次完成。

`ingest_batch_inference.py` 是 adapter hand-off，不是假装替代模型推理的黑盒 runner。YOLO、segmentation、depth 和 HFTF 的具体推理器可以独立批量运行，但必须各自写出同一份 canonical frame JSONL，再由 ingest 绑定 source index、run ID 和输入 hash。

每一行 canonical frame 至少包含：

```json
{
  "schema": "blindassist_candidate_event_mining_frame_v1",
  "source_id": "public-source-a",
  "session_id": "session-001",
  "frame_index": 120,
  "timestamp_ms": 24000,
  "frame_ref": "F:/ba-data/blindassist-candidate-event-mining/public-source-a/frame-000120.jpg",
  "signals": {
    "yolo.coverage": 0.10,
    "segmentation.risk": 0.82,
    "depth.approach": 0.77,
    "motion.crossing": 0.00
  }
}
```

`signals` 必须是 `[0,1]` 的有限归一化分数；单位距离、原始 mask、bbox、depth field 或模型私有 tensor 不进入这个公共接口。具体 adapter 负责在自己的 receipt 中保存原始模型/版本/实现 hash，并把转换后的 canonical trace 交给本 Module。
缺失的 HFTF 或 segmentation 通道必须保持缺失；不能用 `0.0` 伪造“未响应”的负证据。只有实际 sidecar 或明确标注的 proxy 才能写入相应 key。

每个完成的 run 还应在 `F:\ba-data\blindassist-candidate-event-mining\run_index.json` 中登记，绑定 source project index、adapter manifest/trace、全量候选报告、review queue、bundle、Luna receipts 和 candidate pool 的 SHA-256；这只是可追溯索引，不提升任何 evidence authority。

## 输出

每次运行都写入 `artifacts.local/evidence/candidate-event-mining/<run-id>/`：

- `batch_trace.jsonl` 与 `.manifest.json`：已校验的批量推理 hand-off；
- `candidate_report.json` 与 `.sha256`：原始窗口、同 session 去重、跨 session 证据聚类和候选计数；
- `review-bundle/review_bundle_manifest.json`、`review_inputs.jsonl`、`review_prompt.txt`：候选类型、分数、信号 key 和 truth 字段对 Luna 隐藏；
- `luna_reviews.jsonl`：Luna 独立 review receipt；
- `candidate_pool.json` 与 `.sha256`：只收录 `keep` 且达到置信度门的 discovery candidate，其余进入 quarantine。

所有 candidate ID 和 cluster ID 都由输入身份和窗口字段确定性 hash 生成。去重只能合并同一 `source_id × session_id × trigger_type` 的近邻窗口，不能跨 session 合并自然事件；聚类只用于 review 排序/覆盖分析，不改变事件身份。

## 安全边界

- 公开源下载只允许普通公开 URL/API、镜像或归档服务；不得绕过认证、付费、访问控制或技术限制。
- 媒体和下载 receipt 统一放在 `F:\ba-data\blindassist-candidate-event-mining\`；仓库只保存合同、代码、模板和不含原始媒体的证据索引。
- 项目索引至少绑定 `source_url`、`retrieved_at_utc`、`content_sha256`、`media_path` 和 `source_id/session_id`。缺失许可或隐私元数据要记录为未知，不伪造权利；对外共享/再分发另行处理。
- review bundle 的 `candidate_output_visible=false` 是强制字段。Luna 只能看到窗口和完整 taxonomy，不能看到触发类型、分数或其他 review。
- 当前 discovery 允许每个候选一个隔离 `luna_reader` review；低置信、abstain 或缺失 receipt 的候选必须 quarantine。若未来结果成为 claim-critical evidence，不能复用本单路 discovery authority，必须新建双路 review/adjudication contract。
- candidate pool 不接提醒、不改 Kotlin/Android、不改 YOLO/segmentation/depth/HFTF 模型默认值，不写入 production 或安全路径。

## 停止条件

- source/session/frame identity 重复、时间戳不单调、信号非有限或越界、source 未登记、review hash/visibility 漂移：当前 run fail closed。
- 某个 source 无候选是有效的 `zero_candidate_source`，不通过调低阈值静默救援；由下一次版本化 contract 决定是否改变 discovery 假设。
- 去重和聚类不得跨独立 session 合并；窗口过密、持续高频响应或模型通道冲突时保留多个候选并交给 review，而不是覆盖原始触发。
- Luna 无法独立判断、低置信或分歧无法解决时，只 quarantine 该候选，继续处理其他 source；不建立额外外部队列。
- 不因为候选池变大就自动下载更多数据、训练、读取 fresh truth 或启动设备。下一步必须由候选覆盖、重复率、review 清除率和信息增益决定。

## 失败资产复用

候选触发报告可作为 discovery candidate、负例搜索提示、Failure Atlas 入口、review 回归 fixture 或 adapter canary；Luna quarantine 保留为不可评价/需重新设计窗口的诊断。任何 consumed/Development source 仍按原 ancestry 和 evidence role 记录，不能包装成 unseen confirmation。失败的 adapter/schema 版本只关闭该 run/evidence version，修复后使用新的 run ID 和新 hash。
