# GRAIL research module

状态：`active / LAST-METER_ALGORITHM_MAINLINE_REOPENED / PROCEDURAL_M0_MECHANICS_PASS / NATURAL_3D_COVERAGE_GATE_FAIL / STOP_BEFORE_M1`

GRAIL（Goal-Relative Affordance and Interaction Localization）把最后十米重新定义为：给定用户目标，在未见场景中预测一组可到达、目标一致、适合完成交互的 `站立位置 + 朝向`，或显式 `NONE`。

核心分解固定为：

```text
referent != affordance != reachability != visibility != arrival
```

M0 不训练网络。它在 fresh、split-disjoint 的程序化 metric 2.5D 建筑中自动生成目标实例、同类替身、障碍、功能侧和 set-valued interaction pose truth；以 oracle referent + oracle geometry 检查 task/teacher 上界、简单闭环、几何扰动稳定性和四类结构化反事实。

## 稳定 Interface

程序化入口只接收 `--output-dir`；natural-3D 入口只接收冻结的 `--mesh-root`、`--annotation-root`、`--output` 与可选 scene roster。输出状态固定为 set-valued `position + yaw`、`NONE` 或 derived teacher 的 `AMBIGUOUS`，不把 referent、affordance、reachability、visibility、arrival 合并成单一命中。

## 输出

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_m0.py --output-dir artifacts.local/evidence/grail-m0
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_natural_3d_m0.py --mesh-root <mesh-root> --annotation-root <annotation-root> --output <report.json>
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover -s scripts/research/grail -p "test_*.py"
```

程序化结果写入 `artifacts.local/evidence/grail-m0/`。后续 natural-3D runner 使用 ARKitScenes source mesh/OBB 与显式 derived proxy；fresh 结果只有 `20/79` 非空 set，未过 50% coverage 门，故 M1 当前禁止。唯一 successor 必须改变 teacher 信息源，以 source-native navigability/functional-side 或 interaction-pose truth 另立 source-disjoint M0；不得在 fresh cohort 调 proxy。

## 安全边界

M0 是研究 teacher/oracle 证据，不控制真实用户，也不建立 RGB、自然相机、Android、产品或安全 authority。数据集许可、source identity、分母与 proxy/ground-truth 边界必须随结果保留。

## 停止条件

任一预注册 M0 门失败即停止 student/M1；fresh outcome 暴露后不得在同一 cohort 调 threshold、采样距离、类别规则或融合。只有改变 teacher 信息源并建立新的 source-disjoint M0，才可重开 M1 前门。

旧四边界 V1-C/D/E/F 与 passive exact-instance 主线保持关闭。V2-MARKER-POSE 仅作隐藏的 `DEBUG / CALIBRATION / CONTROLLER CANARY`，二维码不进入论文核心或主 Demo。动态风险降为行进过程辅助能力。
