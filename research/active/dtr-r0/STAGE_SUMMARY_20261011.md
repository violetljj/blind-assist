# 10-09 至 10-11：避障线阶段回顾

新数据含新模拟；pp＝百分点；FREE率指采样FREE query误支持率。

## 1. 一页结论

- 【已复现】S在三份新模拟留出增加及时提醒，代价是更多轻提醒；已接受为模拟默认候选，未接入App。
- 【开发观察】DAV2 raw ≥0.8m是RGB比较代表，优势集中中带。
- 【未确认/负结果】Uni近带排序较好，误支持迁移不稳，替换未通过。
- 【未确认/负结果】融合v3去位置、加门控后未确认：ToF有补充信息，但净增小、visit间不稳。
- 【未确认/负结果】ToF退化近带受损，细杆也未稳定解决。
- 【开发观察】候选、记录、manifest可复用；模拟/半合成收益尚未兑现为实机能力。

## 2. ToF：告警级

【已复现】S三份新留出，工作点不同；相对原5格：

| 留出 | 及时净增 | far＋clear有通知片段 |
|---|---:|---|
| [重训次点](nearfield/CNH_TASK_COST_RETRAIN_DEV_20261010.md) | +83/512事件 | far 16→19/256；clear 50→52/512 |
| [确认主点](nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md) | +151/1024事件 | 129→163/1536，+2.214pp |
| [池化点](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md) | +185/1024事件 | 112→155/1536，+2.799pp |

10-10接受取舍：原5格强档逐slot不变，新增均轻提醒；零损由追加结构保证，确认成本两项仍失败。

【开发观察】[14族池化cal](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md)后漂移绝对值1.042→0.195pp，最差新族增3.646pp；工作点/留出也变，非族数因果证明。

【开发观察】[oracle](nearfield/CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)：1.7cm杆前距0.97m、反射率.19/.57时SNR1.37/3.59；0.65m出45°横向视场，非实物上限。[漏检](nearfield/CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010.md)：HEAD未及时38/98事件SNR≥4，均仅一帧；弱、短暂与读出不足并存。

【未确认/负结果】[分级](nearfield/CNH_GRADED_EVIDENCE_DEV_20261010.md)/[走廊峰](nearfield/CNH_GRADED_CORRIDOR_DEV_20261010.md)/[路径](nearfield/CNH_GRADED_MOTION_PATH_DEV_20261010.md)/[稀疏对齐](nearfield/CNH_SPARSE_RAY_TRACK_DEV_20261010.md)/[raw burst](nearfield/CNH_GRADED_RAW_BURST_DEV_20261010.md)：净检出或clear/pass成本不稳，均未升级默认；后三者固定配方已停。

## 3. RGB

【开发观察】[DAV2](nearfield/RGB_DEPTH_BACKBONE_COMPARE_DEV_20261010.md)：三ARK中远见证440/554 POS、误支持25/199 FREE query；远带非全面更好。

【开发观察】[同FREE排序](nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)近带Uni较好；【未确认/负结果】两组六新capture替换失败：[一组](nearfield/RGB_BACKBONE_CONFIRM_DEV_20261010.md)见证3→103/292、误支持0→12/515；[另一组](nearfield/RGB_BAND_HYBRID_DEV_20261010.md)1→48/159、误支持1→29/694＝4.18%>2%上限。分母为POS/FREE query。双模型热态中位0.271秒/帧，仅参考。

【开发观察】换主干后，近带仍受读出尺度/校准限制：部分切点0.27–0.35m>余量上限0.25m。[改读出](nearfield/RGB_NEAR_READOUT_SCALE_DEV_20261010.md)仍损中远；单提高像素精度不够，未排除深度误差。

## 4. 融合：query级，真实RGB＋半合成ToF

【单次确认】[v2](nearfield/SYNC_FUSION_CONFIRM_V2_DEV_20261011.md)中净+385/1923 POS query，95%区间15.02–27.50pp，FREE54/959＝5.63%。【开发观察】[审计](nearfield/SYNC_FUSION_EVIDENCE_AUDIT_DEV_20261011.md)：+132无最低证据；query位置特征由Claude指令引入（用户补充），但不能把+132全归为先验。事后去位置＋门控净+282，非新确认。

【未确认/负结果】[v3](nearfield/SYNC_FUSION_CONFIRM_V3_DEV_20261011.md) D′救213/损126，净+87/2077 POS query，95%区间−0.308～9.155pp，FREE4/689＝0.581%：下界失败。仅ToF类净+209，其余抵消；无证据支持0，属条件拆分。FARO近净−2/273 POS、FREE15/105＝14.286%；ambient×10净−58/1643 POS、FREE121/1702＝7.109%，均失败。

## 5. 数据源

【开发观察】[v1](nearfield/SYNC_RGB_TOF_DATASET_V1_DEV_20261011.md) FARO稀疏，有效配准51/384帧；[v1.1](nearfield/SYNC_RGB_TOF_DATASET_V1_1_DEV_20261011.md)静态sub-ray平均覆盖66.47%；【未确认/负结果】合格仅103/384帧。扰动native仍半循环，FARO有静态/配准/材质/缺测限制。

复用两run的`dataset_manifest_v1.json`/`dataset_manifest_v1_1.json`及来源/覆盖表；派生物随visit隔离。旧eval已消费。

## 6. 开放问题：按影响排序

【开发观察】①sim→real：台架HEAD常越线仅是动机，无正式分母。②[远带FREE](nearfield/FAR_BAND_FREE_DIAGNOSIS_DEV_20261011.md)：未发现错标是共同主因，读出/深度/参考待区分。

【未确认/负结果】③ToF退化稳健性；④同步独立几何覆盖；⑤细杆短暂弱证据。

## 7. 下一步选项：收益是预期，用户选择

| 选项 | 预计收益 | 成本／需要决定 |
|---|---|---|
| 实机台架 | 定位越线/模拟落差 | 高，采集标定；是否进硬件阶段、场景 |
| 远带诊断 | 分清读出/参考缺口 | 低到中，先缓存；是否授权新参考版本 |
| 退化感知融合 | 少被坏ToF拖累 | 中，新配方/visit；FREE取舍、预算 |
| 同步源或细杆 | 扩大验证/弱证据利用 | 高；先补覆盖还是研究细杆 |

## 8. 方法学教训

- 【开发观察】先审位置先验，再解释传感器互补。
- 【未确认/负结果】开发增益缩水；v2/v3模型和训练也变，不能全归过拟合。
- 【开发观察】取舍看收益/打扰曲线，二元成本门不替用户决定。
- 【开发观察】校准背景族少时留意漂移；次数、片段率、FREE与UNKNOWN分开。
