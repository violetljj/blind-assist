# GRAIL M0 Natural-3D Derived Teacher Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`REVERSIBLE_EXPLORATION / DEVELOPMENT_STANDARD / FRESH_SCENE_DISJOINT_ARKITSCENES / DERIVED_FUNCTIONAL_SIDE_AND_NAVMESH_PROXY / VALID_SET_20_OF_79 / ORACLE_POSE_20_OF_20 / ORACLE_CLOSED_LOOP_20_OF_20 / COVERAGE_GATE_FAIL / STOP_BEFORE_M1 / DEFAULT_APP_UNCHANGED`

## 问题与证据合同

程序化 2.5D M0 已证明 set-valued interaction-pose 任务在解析几何中自洽，但不满足“fresh、scene-disjoint 真实 3D scene”的原始 M0 要求。本轮把同一任务迁移到 ARKitScenes 官方自然室内 mesh 与 3DOD instance annotation；不读取 RGB，不训练网络。

Source-native 证据只有：

- metric triangle mesh；
- semantic instance label；
- oriented 3D bounding box。

ARKitScenes 不提供 source-native functional front、navmesh 或 interaction-pose truth。因此 teacher 明确把 floor、最大可走连通分量、身体净空、二维视线和“两个 OBB 候选面中局部合法位姿支持更强的一面”标成 derived proxy；两面支持接近时输出 `AMBIGUOUS/NONE`，不猜正面。

## Adapter Development 与 fresh roster

最初 8 个 V1-F-unused scene 的 4 Development / 4 held-out 诊断全为 `NONE`。检查发现 visibility 错误复用了带 28 cm 身体净空膨胀的 reachability mask，并检查到目标中心，导致每条目标视线在目标表面附近必然被截断。修复后只在 Development scene 上得到 `VALID_SET=4/11`、oracle pose/closed-loop=`4/4`；旧 4 held-out 已消费，不进入最终分母。

fresh roster 在任何 mesh teacher outcome 前按以下条件选定：

- 官方 Training 3DOD metadata；
- 仓库文档、代码和 `DATASET_MASTER_LEDGER.json` 中未出现；
- annotation-only 检查至少含一个 cabinet/shelf/appliance 交互候选；
- 8 个视频来自 8 个互异 visit 或独立 NA scene identity；
- 筛选不读取 mesh 可走空间、face support 或 teacher output。

最终 scene：`40777060 / 40777069 / 40777073 / 40958737 / 40958764 / 41007603 / 40776203 / 41045408`。与 V1-F 的 12 个 scene 交集为 0。

## fresh natural-3D 结果

| 指标 | 结果 | M0 门 | 裁决 |
|---|---:|---:|---|
| held-out scenes | 8 | >=3 | PASS |
| semantic target instances | 79 | >=8 | PASS |
| teacher `VALID_SET` | **20/79 = 25.3%** | >=50% | **FAIL** |
| oracle Interaction Pose Success | **20/20** | 100% nonempty denominator | PASS |
| oracle Closed-Loop Completion | **20/20** | 100% nonempty denominator | PASS |

有效实例按类别为 cabinet 11、shelf 3、oven 2、sink 2、stove 2；`NONE` 为 59/79。按 scene 的有效数为 `0,7,1,5,0,5,2,0`，不是单个 scene 的偶然崩溃。终态：

```text
GRAIL_M0_NATURAL_3D_DERIVED_TEACHER_NOT_ESTABLISHED
STOP_BEFORE_M1
```

这不是 GRAIL 视觉算法负结果：没有训练或运行 student，且 20 个非空 teacher case 的 oracle 闭环全部成功。它否定的是“当前 ARKitScenes mesh + OBB + 局部自由空间 proxy 已足以稳定生成自然 3D interaction-pose teacher”这一前提。

## 裁决与边界

- 不在这 8 个 fresh scene 上修改 floor、clearance、face gap、采样距离或类别规则后重跑晋级；
- 不启动 M1 frozen encoder、B0/B1/B2/GRAIL 训练或比较；
- 不加 Transformer、长期记忆、主动搜索、Android 或主 Demo；
- 程序化 M0 保留为 mechanics regression，不再承担 M1 authorization；
- 旧 exact-instance、四边界与 portal 路线保持关闭，marker-pose 仍是隐藏 canary，动态风险仍为辅助能力。

后续若重开 M1 前门，必须改变 teacher 信息源，而不是在本 cohort 调 proxy：优先使用带 source-native navigability/semantic object geometry 的 Habitat/HM3D 类场景，或具有明确功能侧/交互位姿标注的数据；该动作需要新的 source-disjoint M0 版本。

Claim ceiling：ARKitScenes natural metric mesh + source OBB 上的 derived teacher coverage 与 oracle mechanics；无 source-native affordance/navmesh truth、RGB student、真实相机、自然场景算法泛化、用户、产品或安全证据。默认 App 不变。

## 复现与证据

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_natural_3d_m0.py `
  --mesh-root artifacts.local/evidence/grail-m0/arkitscenes-natural-3d/fresh-mesh-pool `
  --annotation-root artifacts.local/evidence/grail-m0/arkitscenes-natural-3d/fresh-annotation-pool `
  --output artifacts.local/evidence/grail-m0/arkitscenes-natural-3d/fresh-report-v1.json
```

`fresh-report-v1.json` SHA-256：`28F816F96A69F61C309901C9CB790B98788F9C5BD8A394E48394479CD82385E3`。
