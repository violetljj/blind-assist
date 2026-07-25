# Independent person trajectory truth source authority and admission R0 result

状态：`INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT / VALID`

权限：`NO_INDEPENDENT_PERSON_TRAJECTORY_TRUTH_ADMISSION`

## 结论

本轮没有来源同时达到独立性、稳定 person identity、权威 metric 3D、测得的时间同步、完整坐标变换/误差说明和冻结距离分母要求。唯一合法终态是：

`INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT / VALID`

这不是“所有来源都没有外部人体测量”。REveL 与 THÖR 都有独立 mocap，但现有公开/本地证据不足以把它们准入为可与传感器输出配对的独立人体轨迹真值。JRDB 3D box、PCD point-in-box 和 box-conditioned 质心继续硬拒绝为 truth，且本轮没有读取任何候选算法结果、没有比较算法、没有人工挑选轨迹。

## 候选来源判定

| 来源 | 独立测量 | 主要可用事实 | R0 阻塞 | 判定 |
| --- | --- | --- | --- | --- |
| JRDB annotation-derived geometry | 否 | 3D track label、双 PCD、pose/time 可审计 | ROI、identity 与 reference center 仍来自同一 annotation chain；point-in-box 不消除循环论证 | `REJECTED_CIRCULAR_TRUTH` |
| REveL Dynamic Vicon | 是 | 两个 helmet marker 与 sensor-suite marker 共享 Vicon world；已有 20ms 同步审计 | 无本 R0 candidate-blind freeze/独立 validator；无量化 Vicon/外参不确定度；最大距离仅 `10.410m / 6.884m` | `SOURCE_CANDIDATE_LIMITED_NOT_ADMITTED` |
| THÖR people tracks v1 | 是 | Qualisys 100Hz、1mm discretization、平均 residual 2mm；唯一 helmet rigid bodies；共享 world/NTP 说明 | 人工 ID-switch 清理/丢轨恢复无逐帧 provenance；TSV header 无单位；无测得的 Velodyne↔QTM offset/jitter；marker rigid body 到 LiDAR measurement frame 的 lever arm/axes/handedness/extrinsic error 未闭合 | `AUDITED_NOT_ADMITTED` |

THÖR 一手依据为[官方论文](https://arxiv.org/abs/1909.04403)、[people tracks record](https://doi.org/10.5281/zenodo.3382145)和[point-cloud record](https://doi.org/10.5281/zenodo.3405915)。公开论文还说明轨迹曾人工清理 ID switch、恢复丢轨，之后又自动恢复不完整 marker 的位置；本轮没有把这些后处理静默当作无误差真值。

## Candidate-blind freeze 与完整分母

在候选算法输出不可见时冻结：

- source：`thor_people_tracks_v1`；
- member：五个 moving-robot 6D TSV 中按官方 metadata 声明字节数最小的 `Exp_2_run_2_6D.tsv`；
- payload：`40,058,682` bytes，官方 MD5 `17196097564662ca2b72663b8d0a8a3e`；
- window：整个文件；
- person tracks：全部 `Helmet_2..Helmet_10`；
- reference：header 中的 moving-robot rigid body `Citi_1`；
- bands：`0–5 / 5–10 / 10–20 / 20–40 / 40m+`；
- product focus：`5–20m`；
- missing：全零 6D rigid-body record 作为缺失；person 或 reference 缺失只排除对应 person-frame 并守恒计数；禁止插值、换 ID、换 run、截窗或人工挑选。

header 精确给出 `25,912` frame、13 bodies、100Hz，frame `1..25,912` 和 `Time 0.00..259.11s` 严格单调。九条 person track 形成 `233,208` 个 person-frame opportunity：

- valid object-frame：`92,142`；
- missing person：`140,004`；
- missing reference opportunity：`1,062`；
- 守恒：`233,208 = 92,142 + 140,004 + 1,062`。

TSV header 没有单位声明。仅按论文的毫米级说明作 `/1000` **非权威换算假设**时，分带为：

| Band | Provisional object-frame | Distinct tracks | Authority |
| --- | ---: | ---: | --- |
| `0–5m` | `43,821` | `9` | 未准入 |
| `5–10m` | `41,035` | `9` | 未准入 |
| `10–20m` | `7,286` | `9` | 未准入 |
| `20–40m` | `0` | `0` | 空分母 |
| `40m+` | `0` | `0` | 能力边界保留，空分母 |

`5–10` 与 `10–20` 在该换算假设下超过预注册 `>=1,000 object-frame / >=2 track` 数量门，但未通过 metric unit、sensor transform 和同步误差权威，因此不得把 provisional counts 改写成 admitted denominator。`40m+` 没有被删除或并入其他 band。

## Fail-closed gate

通过的只有：

- independent measurement chain；
- candidate-blind freeze。

失败并关闭准入的门：

- stable person track ID authority；
- payload-bound metric 3D unit；
- measured cross-system synchronization error；
- mocap rigid-body 到 sensor measurement frame 的完整 transform；
- recovery/extrinsic/synchronization quantitative uncertainty；
- 可被准入的 product-core distance denominator。

因此不允许用现有 JRDB 标注、LiDAR point-in-box 支持、质心结果、THÖR 后处理轨迹或 REveL seen profile代替独立真值；也不开放 centroid/deskew/tracker/route/event/TTC 比较、Android、人体、独立行走或 production authority。

## 复算与证据

- focused tests：`3/3 OK`；
- independent rebuild validator：`39/39 VALID`；
- config SHA-256：`5450961e2255652b9b90ecb6189f989e6c7c03c101f2e99eaaf366a0e3eb3c51`；
- acquisition SHA-256：`325abffdea873923b44137493e190306a88cc1799913a7e453d8ce655d35f243`；
- denominator ledger SHA-256：`68fb9f3e91eb41a919ad04759f120e36ef1a255c14f9f3fa0fb890f0d04642bf`；
- receipt SHA-256：`0eb4716c91d0af7a90ef9b4d829abe06d7d3d73079cb6b24ee125e6570338647`；
- validation SHA-256：`5ec18e3494080224fc79c30af8380ac81f19d9fadc8aaf990be5d029937b2006`。

机器证据位于 ignored `artifacts.local/evidence/independent-person-trajectory-truth-source-authority-and-admission-r0/`。

## 下一边界

本任务没有自动算法后继。只有新来源或新 source-native authority 同时提供：

1. 原始/逐帧可审计的稳定 person ID 与 recovery mask；
2. 显式 metric unit 和人体 reference-point 定义；
3. person truth 与 sensor frame 的测得时钟 offset/jitter；
4. marker/world 到 sensor measurement frame 的完整外参与误差；
5. candidate-blind 冻结的 `5–20m` 分母，并保留 `20–40 / 40m+` 能力边界；

才可另立新的 source-admission 版本。此前不得进入算法比较。
