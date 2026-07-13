# SANPO 确定性线性 probe 与距离场前置诊断

## 结论

本轮用固定 P1-A 权重中的 **未微调 MobileNetV3 backbone 特征**，在不读取 blind 的前提下执行闭式 ridge 四分类 probe。两次求解的系数和 dev argmax 完全一致，但 OS8 与 OS8+OS32 两个 probe 都未达到预注册可分门，因此当前证据更支持“backbone 表征与 session/split 数据分布不足”，而不是“只要换掉随机 head 初始化即可解决”。

按路线门控，prototype/bootstrap 五组短跑本轮不启动。继续执行它会把不可分表征问题误包装成初始化搜索。

| Probe | 特征维度 | Global mIoU | Boundary IoU | Macro-session | Worst session/scene | 判定 |
|---|---:|---:|---:|---:|---:|---|
| raw OS8 `activation_1` | 96 | 0.2475 | 0.0205 | 0.2348 | 0.0671 | 不可分 |
| raw OS8 + raw OS32 | 672 | 0.3308 | 0.0297 | 0.3279 | 0.0990 | 不可分 |

联合 probe 中最差场景仍为 `step_curb`；其 mIoU 为 `0.0990`、boundary IoU 为 `0.0101`。走路面/障碍/unknown 的联合全局 IoU 分别为 `0.5671/0.2740/0.4526`，说明主要坍塌集中在薄风险边界和对应 held-out session，而不是所有语义都完全不可学习。

## 实验合同

- 数据：`sanpo-v4-real-canonical-r3-20260713`，400 train / 200 dev；trainer 授权报告为 green。
- blind：probe 只解析 canonical `training_manifest.jsonl`，报告固定记录 `blind_holdout_access=not_accessed_by_probe`。
- 特征：P1-A OS8/OS32、alpha 1.0、384 输入；probe 不消费 `lraspp_fuse` 作为主结论，避免用已经训练过的 head 特征证明 backbone 可分。
- 像素：按 sample ID、class 和 flat index 固定排序；每记录每类最多 16 个像素，再按类固定平衡到最多 4096 个。
- 求解：NumPy float64 闭式 ridge，`lambda=1.0`，重复两次。
- 可分门：global mIoU `>=0.35` 且 boundary IoU `>=0.20`。该门只决定是否值得进入 bootstrap 短跑，不是生产晋级门。

本地产物（不提交）：

- `work/codex_sanpo_linear_probe_20260713/backbone_os8_probe.json`，SHA256 `66fc5420b8160d314efd18eac09f6899ad69faf52c2381e9e05a793cda964cd8`
- `work/codex_sanpo_linear_probe_20260713/backbone_os8_os32_probe.json`，SHA256 `a6e976033333068341f987563d39b74a1c832ed647d3208da916920f1a8f37e9`

## 距离场辅助监督前置诊断

`sanpo_boundary_distance_aux.py` 已独立实现 boundary/step/curb 截断 signed/unsigned distance target、空 mask/全 mask sentinel、SmoothL1 权重和 blind 拒绝合同。它尚未接入 trainer 或模型图。

在训练实际的 384×384 nearest-resized mask、signed distance、截断半径 16 像素下：

| Split | Frames / sessions | 有 boundary 的帧 | Boundary 像素占比 | `abs(target)<1` 近边界占比 |
|---|---:|---:|---:|---:|
| train | 400 / 8 | 293 | 0.857% | 3.839% |
| dev | 200 / 4 | 145 | 16.976% | 8.059% |

dev boundary 像素占比约为 train 的 `19.8×`。因此现在直接比较“有/无距离辅助损失”会同时受到 split 分布错位影响，不能把结果诚实归因给距离场。距离头保留为独立合同；等 P3 session/split 重构闭合后，再以相同五组 seed 做 OFAT 训练。

诊断报告：`work/codex_sanpo_distance_aux_20260713/report.json`，SHA256 `fe329dceeae653b75890fdac9753e88ae51539949a95698229cbf70b3e0240a4`。

## 决策

1. 停止 prototype/bootstrap 五组短跑：特征可分前置门未通过。
2. 不把距离场接入当前 trainer：先关闭 P3 session/split 的 `19.8×` boundary 分布差。
3. 已启动反事实 episode 采集合同；它将以 matched positive/negative 和 LOSO 事件指标补足像素监督。
4. 中期风险轮廓 + 生命周期头、SAM/ASAM 继续保留在 `idea.md`，但不得绕过上述根因门。
5. 所有结果保持 `benchmark-only`、`do_not_replace_default_model`；不导出 INT8，不运行设备门，不修改 App 模型。
