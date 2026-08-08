# BlindAssist Assistive Geometry program route decision

状态：`ROUTE_SELECTED / B0_CONTRACT_NOT_FROZEN / NO_TRAINING_AUTHORITY / DEFAULT_APP_UNCHANGED`

日期：`2026-08-09`

## 1. 决策

项目算法主线从“把 DepthART-S 部署为 DA2 的替代 backbone”提升为：

> **构建面向助盲通行空间的轻量 Assistive Geometry Model。**

DepthART-S 保留为优先 encoder/initialization、depth baseline 和部署研究载体，但不再定义
项目算法贡献的上限。DA2 继续保持冻结的 teacher、baseline、regression reference 和 fallback。

执行顺序不再是严格串行的 `A 部署完成 → B 自有模型`。新的顺序是：

```text
B0 任务/数据/几何合同
       ├── B1/B2 Assistive Geometry 算法主线
       └── A-D1 task-preserving 部署使能线
                    ↓
             共同 GeometryState 合同
                    ↓
C 异质教师增强（通过互补性门后）
                    ↓
D Memory / Future / Uncertainty / Adaptive Compute
```

部署支线不能阻塞算法问题的提出；算法结果也不能反向修复 strict G4-D 或自动授权部署。

## 2. 三份设计材料的当前角色

| 材料 | 当前角色 | 可继承内容 | 不可作为当前真源的内容 |
|---|---|---|---|
| 《2024–2026 新型轻量智能学习与实时视觉感知技术路线整理》 | 长期系统创新地图 | streaming memory、future geometry、uncertainty、dynamic compute | 论文数字、技术成熟度和当前执行权限 |
| 《BlindAssist DA3 × Metric3D GeoMetric-Lite R0.1》 | 训练增强与教师互补性设计库 | metric/geometry 职责解耦、teacher disagreement、clearance-aware objective、false-clear 非对称惩罚 | 自动启动双教师训练或把 pseudo-label 当真实安全 truth |
| 《DepthART R1 QAIRT/QNN/HTP 部署与准入执行纲领》 | 历史基础设施快照 | DepthART encoder、Ground/Confidence/Clearance heads、任务级准入思想 | Snapdragon 8 Gen 2、Acos frontier、SelectiveScan 未触达和当时 HTP 状态 |

第三份材料中的平台状态已被仓库内 SM-S9280 / SM8650 / Snapdragon 8 Gen 3 / HTP v75
实测替代。以后只能引用其设计思想，不能引用其“当前状态”。

## 3. 三个互不替代的科学问题

1. **Strict tensor parity**：PyTorch/ONNX/HTP raw depth 是否满足冻结的逐元素数值合同。
   当前 strict G4-D 负终态永久保留。
2. **Task-preserving deployment**：设备输出差异是否保持 clearance、false-clear、false-block、
   temporal 与 critical decision。当前 D0 没有合格 precision arm；D1 尚未激活。
3. **Assistive geometry learning**：任务特定表示和目标是否优于 depth-only backbone + 固定后处理。
   这是新的算法主线，不能借部署结果或 teacher pseudo-label 自动获得 admission。

任何一个问题的 PASS 都不能改写另两个问题的 FAIL、NOT_EVALUATED 或权限边界。

## 4. 模型边界

暂定研究标识：

```text
BLINDASSIST_ASSISTIVE_GEOMETRY_R0
```

论文描述暂用：

```text
DepthART-initialized BlindAssist Assistive Geometry Model
```

“DepthART-initialized”是初始化/候选实现，不是永久架构约束。共享 encoder 必须允许在相同
输入输出合同下替换，防止研究退化为 DepthART 的附属变体。

建议的逻辑接口为：

```text
RGB + K + transform receipt
            ↓
replaceable lightweight encoder
            ↓
GeometryState
  ├─ dense_depth_m + depth_validity       # auxiliary，可在后期降频或裁剪
  ├─ ground_probability / ground_model
  ├─ clearance_m[band, horizon]
  ├─ body_swept_occupancy[band, horizon]
  ├─ confidence[task]
  └─ unknown_mask + unknown_reason
```

`GeometryState` 还必须绑定 frame/session identity、capture timestamp、原始 K、resize/crop/pad
后的 K、FOV/有效像素区域和 provenance。缺失、过期或不支持的几何不能被填成 `CLEAR`。

## 5. 算法贡献最低要求

仅增加四个 head 不足以构成强论文贡献。至少要通过预冻结消融证明以下组合中的实质贡献：

- 身体扫掠包络和近场通行空间的任务表示；
- truth-bound clearance objective；
- 对 truth-occupied / predicted-clear 的非对称 false-clear 惩罚；
- confidence calibration 与显式 UNKNOWN，而不是把低置信样本当负类；
- 在互补性先验成立后，metric teacher 与 geometry/temporal teacher 的职责路由；
- 任务质量、延迟、功耗和内存的设备 Pareto，而非只比较 AbsRel 或参数量。

Teacher depth 导出的 clearance 可以用于蒸馏，但只能证明 teacher preservation。实际 false-clear、
false-block 和通行判断必须绑定独立 truth/有效几何；pseudo-label 不能产生 safety authority。

## 6. 分阶段路线

| 阶段 | 研究问题 | 进入条件 | 产出与权限 |
|---|---|---|---|
| B0 | 任务、相机、身体几何、数据和评价合同是什么 | 本路线决策 | 仅设计合同；不训练、不看独立 outcome |
| B1 | DepthART-initialized 多头基线是否可学 | B0 完整冻结，Development roster 独立性成立 | Development-only baseline；不替换 DA2 |
| B2 | 哪些 task head/loss 真正改善通行几何 | B1 可复现 | depth-only、+ground、+clearance、+false-clear、+confidence 的逐项消融 |
| A-D1 | fixed-mixed HTP 是否保持 B0 任务合同 | 产品纵横比/K/truth/postprocess 与新 roster 冻结 | 仅选出 R2 candidate；strict G4-D 不变 |
| C0 | DA3 与 Metric3D 是否有可利用互补性 | 新的 teacher-evaluation cohort 与真值合同 | Oracle/互补性/分歧相关性结果；仍不训练双教师 student |
| C1 | 异质教师是否提升 B2 student | C0 通过预冻结 kill gate | Development-only 双教师增强候选 |
| D | 记忆、未来几何和动态计算是否有增益 | GeometryState 和单帧基线稳定 | GRU/TCN/SSM、future clearance/TTC、compute gate 的独立消融 |

时序结构不预先指定为 Mamba/SSM。GRU、TCN、SSM 必须在同一 GeometryState、任务门和设备预算
下比较；模型名称不能替代实测收益。

## 7. B0 唯一 successor

下一步只允许建立：

```text
BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT
```

B0 必须在任何新 student 训练和 candidate outcome 前冻结：

1. 产品输入纵横比、分辨率候选、FOV、resize/crop/pad 和 K 传播；
2. 身体宽度、高度、余量、band/horizon 与 clearance/occupancy 的数学定义；
3. `CLEAR / OCCUPIED / UNKNOWN` 及每类 `unknown_reason`；
4. depth、ground、clearance、confidence 的 truth、teacher、pseudo-label 和 diagnostic 角色；
5. parent/session/capture 级切分、近重复审计和已消费 cohort 排除；
6. DA2 depth-only reference、DepthART depth-only baseline 与 Assistive Geometry arms；
7. 消融顺序、随机种子、checkpoint 选择规则和失败停止门；
8. clearance MAE、false-clear、false-block、coverage、confidence calibration、temporal 与
   critical-decision 指标的分母和绝对/非劣门；
9. B0 与 DepthART D1 共用的 task postprocess 版本和 receipt；
10. 独立确认 cohort 的身份只可 metadata-lock，不得参与训练、调参或选模。

当前未冻结的数值必须写成 `UNRESOLVED` 并阻塞后续阶段，不能由实现者静默选择。

## 8. 数据与证据边界

- 已消费的 120-frame/既有 Development outcome 不得重新切分、调参或选择 B1/B2 arm。
- 8 个 ARKitScenes R2 session 保持 metadata-only、payload 未打开，不得参与 B0/B1/B2 选模。
- Metric3D、DA3、DA2 或 DepthART 输出都属于 model evidence，不是现实通行安全真值。
- 缺少 ground/camera-height/body-envelope 对齐的样本，只能对可支持的 claim 有效；`UNKNOWN`
  不是 negative。
- B1/B2/C 的训练、选择和独立确认必须使用预先分离的数据角色和新 roster。

## 9. 部署支线的当前边界

保持以下不可变结论：

```text
CURRENT_QAIRT_2_47_SM8650_HTP_STANDARD_FLOAT_PATH_STRICT_G4D_NOT_SUPPORTED
```

D0 三条 precision recipe 无合格 arm，未进入任务 outcome 或性能；fixed-mixed `448x448` 仍只是
数值 canary/control。A-D1 只有在 B0 冻结产品几何和 task postprocess 后才能另立，且通过后
只允许锁定一个 R2 candidate。禁止测 latency 来绕过质量前门，也禁止把 B/C 结果回写为旧
G4-D PASS。

## 10. 停止门

- B1 若不能在 truth-bound clearance/false-clear 上优于 depth-only baseline，停止堆叠复杂 teacher。
- C0 若 Oracle 双教师不优于最佳单教师、DA3 时序优势不成立，或 disagreement 与真实 error
  无稳定关系，则关闭对应双教师主张。
- D 若轻量时序在 parent/session-disjoint 评价中无增益，保留单帧 GeometryState，不因技术新颖
  强行引入 SSM。
- 任一平均指标改善若依赖 coverage 塌缩、UNKNOWN→CLEAR、独立性破坏或 outcome 后调门，判 FAIL。

## 11. 当前允许与禁止

允许：撰写和审查 B0 合同、盘点未消费数据与 label capability、建立无 outcome 的 schema/tests。

禁止：训练 student、运行双教师全套、读取独立 confirmation outcome、修改 D0、激活 R2、替换
DA2、修改默认 App、作 production 或 safety 声明。
