# S集成1.25×预算新留出确认协议（渲染前提交）

2026-10-10。实验/协议ID：`cnh-s-ensemble-confirm-dev-20261010`；冻结父结果`6e441f9a`。

依据[WORKFLOW](../../../WORKFLOW.md#work-phases)的Confirm行，本任务检验固定方法和固定比较的预声明主张，采用登记与渲染前提交协议的最低正式确认通道。证据域仍为新生成模拟Development；不访问保护480/test，不自动替换App或主线。S集成及1.25工作点曾由已消费Development次点结果提名，此选择披露，不把旧结果当确认数据。

## 身份与新数据

[PLAN](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/PLAN.json) SHA256 `7f209b8e99cd6ed3ef9d7ea916576d89e4fb4c4fe78f136ee4bbf94fb28e01a5`；[源协议快照](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/protocol_snapshot.md) SHA256 `c5728bb891c271a7c84f7228ec9e7cc9aa0609c80a92297266119ac3a22725ab`；[frozen_spec](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/frozen_spec.json) SHA256 `389690ecdb86828dc4b38aebb43e36ed6a2bf593695cff93b6767e23c8b07ca9`。PLAN与全部scene参数先保存，仅完成作者几何/支持，不采光子或运行模型。场景SHA `f6d44ef39df2cb0470cc8141e8f9e8b277d42b7fdb63bc754aa2c74d7ef00fa8`；库存SHA `8fd22ba267ad82423a293d53c93d3d4c18944570caf9d7cc0a3662f00574cc05`。

cal768 physical scene×K2（2新family、4背景实例）：contact每高度128scene，pass四层每层64scene，clear256。hold1536×K2（4新family、8背景实例）：contact每高度256scene，pass每层128scene，clear512。比重同重训run。横杆/竖杆/标牌边缘/突出物、双尺寸/反射率、双侧均覆盖；暗薄定义沿用父run。

与27份可访问已列AABB元数据、16936历史物理键（含重训train/cal/hold）核对：物理键、背景几何/family、目标轴向尺寸/厚度/反射率零交集；新cal/hold互不交集。未读取旧观测/结局选点。无同构AABB键的实录/UE及保护来源只列身份，不声称全历史世界排除。审计范围见[inventory](../../../../artifacts.local/work/cnh-s-ensemble-confirm-dev-20261010/inventory.json)。六family名称和具体box全部在PLAN。

冻结数值参数（m；rho无量纲）：

```json
{
  "cal": {
    "width": [
      0.38917,
      0.84917
    ],
    "thickness": [
      0.01417,
      0.09417
    ],
    "rho": [
      0.16117,
      0.64117
    ],
    "edge": [
      0.03017,
      0.05417
    ],
    "protrusion_width": [
      0.19417,
      0.27417
    ],
    "protrusion_depth": [
      0.07417,
      0.11417
    ],
    "protrusion_halfheight": [
      0.07917,
      0.11917
    ],
    "vertical_height": [
      0.35017,
      0.36017
    ],
    "sign_halfheight": 0.07917,
    "contact_inner": [
      0.28017,
      0.22017
    ],
    "pass_gap": [
      0.02417,
      0.07417,
      0.16417,
      0.27417
    ],
    "clear_gap": [
      0.44417,
      0.56417
    ]
  },
  "hold": {
    "width": [
      0.46923,
      0.92923
    ],
    "thickness": [
      0.02023,
      0.10423
    ],
    "rho": [
      0.24123,
      0.58123
    ],
    "edge": [
      0.04023,
      0.06423
    ],
    "protrusion_width": [
      0.22423,
      0.31423
    ],
    "protrusion_depth": [
      0.08423,
      0.12423
    ],
    "protrusion_halfheight": [
      0.08023,
      0.12023
    ],
    "vertical_height": [
      0.35223,
      0.36223
    ],
    "sign_halfheight": 0.08023,
    "contact_inner": [
      0.27023,
      0.21023
    ],
    "pass_gap": [
      0.03423,
      0.08423,
      0.18423,
      0.32423
    ],
    "clear_gap": [
      0.48423,
      0.60423
    ]
  }
}
```

## 冻结方法、输入与比较

S955/956/957每高度6个HGB逐字节复用，100轮等父配方不变；**无训练**。每seed使用对应ordinary的平滑margin、趋势斜率/残差3项＋current峰22项＋无效性22项=47维；直接执行冻结父controller `features_and_labels` 函数，模型输入不增删。S集成采用父实现float32三seed评分均值后存档，不改精度。M3/local仅固定原5格强档，joint-parent用于E门控。作者几何、类别、尺寸和target_support仅用于生成/标签诊断、评价分组，不进入模型。

原5格strong逐slot锁定，新增仅在非strong slot追加light，强转轻0。gap1逐query通知器和同帧最高等级联合计数不变；contact f3–13及时、f14–15晚、全窗无通知静默。原M3、原5格、原both955作固定对照；E三ordinary均值沿用父门控max(single margin,joint-parent margin)，同规则新cal1.25选点。

| 权重/实现 | SHA256 |
| --- | --- |
| cnh_task_cost_retrain_20261010.py | b732c500f857741ab754a0ebbcc133f7b5a823360b7cba00acc310e2f4a05f2a |
| cnh_task_cost_train_20261010.py | 8b203d0204090c5958d2ebdc3b8fb16dc4a6206ba78863283969fea69aa971b3 |
| cnh_task_cost_metrics_20261010.py | b68764d994ef60ed8f08c541a7a3ee8135fd4190d677b61cad9c8ac1d341e54d |
| S_2026100955_BODY.pickle | 824a93352542c6f8648e864385b9bae8c9c31194bd265e59ddabbe6a6d59a196 |
| S_2026100955_HEAD.pickle | b588beef916dbf052b945d602c7ace7720c163b0460e6aab4fbdcbdad7af82b3 |
| S_2026100956_BODY.pickle | dc0804a9693a4488055bc86fa21a25aa7e28bec44fb35104e1002595ce049b96 |
| S_2026100956_HEAD.pickle | 38f95121a15a56c07a1c64103c83feacc705a89428e5bf5efdfcc8ad22952071 |
| S_2026100957_BODY.pickle | 7b7812b893bb85518da9fd66f89c1ebac6518581ad8608b3bf0e814135724e60 |
| S_2026100957_HEAD.pickle | dc968fe2853eca555960b233e13997df635b471304067e73cdf5b0b9354077d2 |
| cnh_s_ensemble_confirm_source_20261010.py | a34aec437ca0b67ef047bfd47c8ff95f46d3e3f788f51c561715022188b91382 |
| cnh_s_ensemble_confirm_metrics_20261010.py | 9650995bec7fd0e6131b59a629f7e758a28e2d9616c8c3ac6acdf19a31efc08a |
| cnh_s_ensemble_confirm_20261010.py | f05625db04d5356718bcd68078d3209bd7d543c00455acd155b56c832311c025 |

冻结依赖包及旧阈值全hash在frozen_spec（继承父E2E冻结清单）。渲染前、cal后、hold后核对身份；执行manifest记录本协议commit及代码hash。独立审计源码可做审计自身机械修复，不能修改科学数据/模型/特征/评价规则。

## 唯一主工作点与四项判据

v2成本：最近表面到共向±.30m走廊边缘间隙≤.10m为near-pass；light=.25、strong=1。far-pass/clear任何通知=1。联合加权成本为负例通知权重之和；0/.5仅敏感性描述。

新cal仅负例评分ties全部枚举，包含-inf/nextafter每个有限负例tie/+inf；实际gap1成本逐一算，不假设单调，选成本≤1.25×原5格cal成本的最低阈值。contact收益不用于选点；S3单seed及E同规则。完成cal切点/hash封存后才渲染、前向、读取hold。辅助点只迁移父封存S集成次点**精确tau=2.4142577648162846**（展示2.41426），只描述漂移，不用于主判定。

hold每高度512contact事件（256physical clusters）；合计1024事件（512clusters）。全体四项AND：

1. (a) S集成主点 timely净增≥80/1024=7.8125%（父40/512等比）；HEAD、BODY各≥ceil(512×.02)=11件。
2. (b) far+clear满额通知≤原5格hold×1.10，保留实数上限，整数通知不得越界。
3. (c) hold加权成本≤原5格hold×1.25×1.05=1.3125倍。
4. (d) S955/956/957分别按自身cal同规则切点，hold总净增均≥40/1024。

通过写“S 集成 @1.25× 建议升为 ToF 默认候选”，并列出实际主线升级所需的正式inheritance/部署验证步骤；本run只推荐，不替换现任。未通过保留原5格，逐项报告失败与混合收益。全分母、暗薄/标牌边缘、近远各层强轻次数及有通知clip/physical-scene率完整描述，不择子集。无新增hold阈值选择/方法搜索。

## 决策精度、执行和停止

父次点提名是83/512净增，本run检验最低40/512的实用门槛，且必须通过负例成本/seed约束；分母加倍不是把K/帧当独立样本。单个contactscene K2的救回比例为0/.5/1；零损由强档锁定结构性保证，非统计非劣证明。条件性独立scene假设下最坏SD≤.5，总512scene的SE≤2.21个百分点、每高度256scene≤3.125个百分点；这里只作采集前精度量级，系统性factorial场景与仅4holdfamily不支持真实总体置信/硬件结论。gate采用预声明工程计数，不声称显著性或安全保证。

渲染+冻结前向GPU≤1200command-wall s；评价/准备/源审计/整合≤1200command-wall s。S的CPU预测/特征开销同时保守记入阶段预算；GPU科学顺序执行避免争用。到任何上限立即停止该阶段，报告partial/NOT_EVALUABLE，不换源或补样本。所有run不训练。

工程失败可在同源、同seed、同模型及规则下恢复未完成阶段；只复用完成且hash核验的输出；原失败与部分产物保留，绝不覆盖收据。模型/几何/门槛变动需要新用户授权协议，不能让本run事后通过。若渲染/模型call状态不明，记in_doubt且已耗时入账；确认缓存身份才续跑，不重采seed。完整声明分母未达不能宣布四gate通过。CPU/GPUparity失败则停科学评价，不当方法否定。

独立核验采用另一标量gap1/tie实现及源/权重身份、逐ledger/聚类核对，预算内优先主判据。任务资源在finally释放；共享hardware任务不动。输出均在canonical artifacts目录，登记与协议在渲染前normal commit/push master，最终更新CURRENT/RUNS/CURRENT_DECISION并关闭登记。

## 继承与主张上限

冻结S组件责任为“原5格非strong slot的任务成本light排序”，该版本在模拟新family中的候选贡献；保留原5格RETAINED_CORE角色及旧run失败/停止。此确认不改原S同成本主点失败。若通过，角色只为待主线整合的COMPONENT_OR_CHALLENGER/COMPONENT建议；正式主线或受管复用改变还需对应terminal inheritance登记与界面/资源/通知器实机验证，依证据域另行评估。失败亦不推出所有特征理论不可分。新hold消费后不再作新确认，保护480/test无访问。真实分布、硬件上限、安全、完整RGB/CNH同步不在本run主张内。
