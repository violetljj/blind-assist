# USTRF-SC 的 SANPO 数据回放接入

## 当前结论

新增 `scripts/acquire_sanpo_synthetic_replay.py`，将官方 SANPO-Synthetic 的最小连续窗口接入为可复现的离线回放包。每一帧绑定 RGB、官方 panoptic mask、官方 metric-depth 原始字节、相机位姿、相机内参、official train split 与 GCS MD5/SHA256 receipt。缺失的 IMU 不得以推测或插值补造，而是明确标为不可用于回放。

该工作只完成数据入口；它不训练模型、不改变 Android 默认 YOLO、不产生用户可执行导航指令，也不把像素级来源标注写成助盲事件真值。

## 运行

```powershell
.\.venv-export312\Scripts\python.exe scripts\acquire_sanpo_synthetic_replay.py `
  --frame-count 3 `
  --output-root test-artifacts.local\datasets\sanpo-synthetic-replay-20260720
```

输出包含 `manifest.replay.jsonl`、`dataset_spec.json`、每个原始模态文件和 `qa/replay_validation.json`。目录拒绝覆盖，原始数据保持在 Git 忽略的 `test-artifacts.local`。

对完整窗口运行可重复的预训练入口审计：

```powershell
.\.venv-export312\Scripts\python.exe scripts\audit_sanpo_synthetic_replay.py `
  --replay-root artifacts.local\evidence\datasets\sanpo-synthetic-replay-25frames-20260720 `
  --report artifacts.local\evidence\datasets\sanpo-synthetic-replay-25frames-20260720\qa\pretraining_intake_audit.json
```

审计要求每个 RGB、mask、depth 文件仍与 manifest SHA256 一致，所有原始 SANPO 类都能以共享 `SANPO_MAP` 映射，并且窗口中覆盖四类目标语义。通过结果只表示可成为 benchmark-only 预训练候选。

再执行 raw metric-depth 结构审计：

```powershell
.\.venv-export312\Scripts\python.exe scripts\audit_sanpo_synthetic_metric_replay.py `
  --replay-root artifacts.local\evidence\datasets\sanpo-synthetic-replay-25frames-20260720 `
  --report artifacts.local\evidence\datasets\sanpo-synthetic-replay-25frames-20260720\qa\metric_replay_audit_20260720.json
```

本次 25 帧窗口通过：每个 depth 文件的 float16 头部均声明为 `2208×1242`，且 payload 精确为 `2,742,336` 样本、全部含有限正深度。结果仅说明官方 metric-depth 原始字节可作为 **离线输入完整性** evidence；不会判断地面、台阶、可通行性或任何助盲事件。

`camera_poses.csv` 虽有足够行数覆盖请求的 source index，但列中没有显式 frame index 或 timestamp。因此审计明确写出 `ustrf_pose_warp_admitted=false`：不能以行号推断替代精确 frame/time 绑定，也不能把 dataset camera pose 扩大为独立验证的人体/设备 body-frame receipt。

`core:ustrf` 现有 `UstrfVerifiedPoseDelta` 已由 `UstrfSafetySession` 消费：receipt 必须严格绑定相邻 frame、显式标记为独立验证，才会把静态风险从上一用户局部栅格重投影到当前栅格；任何错误会 reset field 并留下 fail-closed STOP record。SANPO 相机位姿文件可作为未来 desktop replay adapter 的来源，但在完成坐标轴/外参和 frame 对齐核验前，不能把数据集 pose 直接标为已验证 receipt，更不能接入手机安全快环。

## 数据顺序与门禁

1. SANPO-Synthetic 只可作为 benchmark-only 预训练候选；先验证四类像素映射与离线回放，再考虑训练。
2. SANPO-Real 的真实连续序列仍独立承担微调/开发/盲集隔离；既有 canonical recipe 与 review/finalize 门禁不被本脚本修改。
3. 公开视频只可作为许可、脱敏、哈希和银标隔离后的补充压力测试；它不能填补 Real 的事件真值，也不能改变训练/校准/blind/默认模型权限。
4. 任一候选还必须分别通过离线、INT8 与同设备连续事件门；任一未通过即维持 `do_not_replace_default_model`。

官方 SANPO 同时提供真实与合成的第一视角 RGB、分割、深度与相机相关数据；合成会话的发布清单在本次最小样本中未列出 IMU 文件，因此接入器 fail-closed 地记录此缺口，而不是假定 IMU 可用。[SANPO 官方数据说明](https://github.com/google-research-datasets/sanpo_dataset)
