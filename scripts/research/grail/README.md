# GRAIL research module

状态：`active / M1_REFERENCE_ONLY_STOPPED / GRAIL_R0_RELATIONAL_ORACLE_PASS / R1_SIGNATURE_OBSERVABILITY_COMPLETE / R1_OBTAINABLE_RELATION_ACQUISITION_ONLY / STOP_BEFORE_M2`

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
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/freeze_grail_procthor_native_m0.py --dataset <test.jsonl.gz> --docker-image-id <sha256:...> --output <manifest.json>
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/freeze_grail_m1.py --val <val.jsonl.gz> --test <test.jsonl.gz> --output <manifest.json>
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_relational_r0.py --dataset <val.jsonl.gz> --collection <v2b-dev-collection.json> --features <v2b-features-dev.pt> --checkpoint <v2b-checkpoint.pt> --development-result <v2b-development-result.json> --output <relational-oracle-result.json>
E:\codex-tools\bin\blindassist-python.cmd scripts/research/grail/run_grail_relational_observability_r1.py --dataset <val.jsonl.gz> --collection <v2b-dev-collection.json> --features <v2b-features-dev.pt> --checkpoint <v2b-checkpoint.pt> --development-result <v2b-development-result.json> --r0-result <relational-oracle-result.json> --output <signature-observability-ablation.json>
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover -s scripts/research/grail -p "test_*.py"
```

程序化结果写入 `artifacts.local/evidence/grail-m0/`。ARKitScenes source mesh/OBB derived proxy 的 fresh 结果只有 `20/79` 非空 set，未过 50% coverage 门，该信息源保持关闭，不得在原 cohort 调 proxy。

ProcTHOR native M0 V1 因空 position precondition 未被 runner 显式映射为 `NONE` 而在首个 house 终止，状态为 `NOT_EVALUABLE`，不得重跑或解释。V2 以新 salt 和全新 roster 冻结后唯一执行，12 scenes、205 targets、7 types；pose coverage=`199/205`，oracle pose/path=`199/199`，local stability=`191/199`，action canary=`12/12`，NONE false commit=`0/18`，counterfactual=`572/572`。全部门通过，故只在 synthetic/native claim ceiling 内建立 M0 upper bound 并授权 M1。

M1 已在任何视觉 collection/outcome 前冻结 24 train / 6 dev / 12 test houses、DINOv2-S encoder、Depth-Anything-V2-S B1 evidence、B0/B1/B2/GRAIL interface 与 one-shot gates；test houses 与全部已消费 M0 test houses 分离。

M1 V1 Development 因 query target-centering leak 在 formal test 前关闭。V2b 用 hash-ranked off-center visible yaw 重建 418 train / 78 dev positives；GRAIL pose=`22/78`、wrong-target=`16/43`、absence false commit=`3/78`、permutation=`78/78`，未超过最强 B1 pose=`23/78`。因此 formal test 保持未打开，当前 `STOP_BEFORE_M2`。

GRAIL-R0 在同一已消费 Development 78-case 上冻结 candidate set、checkpoint、pose head、threshold 与 evaluator，只增加不含 object ID 的 ProcTHOR native coarse relation signature。referent top-1=`75/78`、complete pose=`57/78`、wrong-target=`0/43`、absence false commit=`0/78`、complete rescue/collateral=`35/0`。该 privileged-metadata oracle 只建立“独立关系信息可以击穿 bottleneck”的机制上界；唯一 successor 是 R1 可获得关系表示，不授权 M2 或 formal test。

R1 signature observability ablation 继续复用同一 consumed Development，逐组投影 R0 signature。`semantic type + native root/part sibling ordinal + nearest stable object type` 已完整复现 R0 的 `75/78` referent、`57/78` complete、`0/43` wrong-target 与 `0/78` absence false commit；去掉 sibling ordinal 后仅 `48/78`、`31/78`。方向、距离、相对高度、support、room 和 coarse height 在该 cohort 上不是必要字段。这个结果只收窄 student 输入目标；root/part grouping 与 relation 仍来自 privileged metadata，尚未证明 RGB/text 可恢复。

## 安全边界

M0 是研究 teacher/oracle 证据，不控制真实用户，也不建立 RGB、自然相机、Android、产品或安全 authority。数据集许可、source identity、分母与 proxy/ground-truth 边界必须随结果保留。

## 停止条件

任一预注册 M0 门失败即停止 student/M1；fresh outcome 暴露后不得在同一 cohort 调 threshold、采样距离、类别规则或融合。只有改变 teacher 信息源并建立新的 source-disjoint M0，才可重开 M1 前门。

旧四边界 V1-C/D/E/F 与 passive exact-instance 主线保持关闭。V2-MARKER-POSE 仅作隐藏的 `DEBUG / CALIBRATION / CONTROLLER CANARY`，二维码不进入论文核心或主 Demo。动态风险降为行进过程辅助能力。
