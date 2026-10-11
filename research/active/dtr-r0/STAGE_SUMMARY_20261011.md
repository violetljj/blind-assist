# 10-09 至 10-11：避障线阶段总结

新数据可含新模拟；pp＝百分点，W/F分别为POS/FREE query见证/误支持；UNKNOWN非负例。

## 1. 一页结论

- 【已复现】S以轻提醒换及时收益，保留ToF模拟默认候选。
- 【开发观察】DAV2 raw用于≥0.8m，优势集中中带。
- 【未确认/负结果】Uni近带替换失败，排序优势未兑现为稳定工作点。
- 【预注册确认】冻结D′中带低误报排序胜RGB/OR，部署点非双向占优。
- 【未确认/负结果】近带融合、ToF退化与细杆未解决，远带仍待查。
- 【开发观察】manifest可复用；半合成收益不等于实机/告警事件收益。
- 【未确认/负结果】sim→real未闭环，先诊断已有真实录制。

## 2. ToF：告警级

【已复现】不同工作点的三份新模拟留出，及时收益一致：

| 留出 | 及时净增／事件 | far＋clear有通知片段 |
|---|---:|---|
| [重训次点](nearfield/CNH_TASK_COST_RETRAIN_DEV_20261010.md) | +83/512 | far16→19/256；clear50→52/512 |
| [确认主点](nearfield/CNH_S_ENSEMBLE_CONFIRM_DEV_20261010.md) | +151/1024 | 129→163/1536，+2.214pp |
| [池化点](nearfield/CNH_S_ENSEMBLE_TRADEOFF_DEV_20261010.md) | +185/1024 | 112→155/1536，+2.799pp |

10-10接受取舍：强档逐slot不动，新增均轻提醒；成本确认失败保留，未接App。

【开发观察】14族池化后，片段率增量漂移绝对值1.042→0.195pp；最差新族增3.646pp。非族数单独因果证明。

【开发观察】[细杆](nearfield/CNH_HEAD_THIN_SIGNAL_ORACLE_DEV_20261010.md)1.7cm、前距0.97m，rho=.19/.57时最佳bin SNR中位1.37/3.59；[HEAD漏检](nearfield/CNH_BASELINE_MISS_DECOMPOSITION_DEV_20261010.md)38/98事件SNR≥4，均仅一帧。信号与读出瓶颈并存，非实物上限。

【未确认/负结果】具体配方的去留：

- [分级静默](nearfield/CNH_GRADED_EVIDENCE_DEV_20261010.md)：丢contact，未采用。
- [走廊峰过滤](nearfield/CNH_GRADED_CORRIDOR_DEV_20261010.md)：BODY/clear取舍不稳，未采用。
- [运动/静态路径](nearfield/CNH_GRADED_MOTION_PATH_DEV_20261010.md)：无稳定及时收益，配方停。
- [稀疏对齐](nearfield/CNH_SPARSE_RAY_TRACK_DEV_20261010.md)：同通知预算无救回，配方停。
- [raw burst](nearfield/CNH_GRADED_RAW_BURST_DEV_20261010.md)：2-of-3停；single代价高，留对照。

## 3. RGB

【开发观察】[DAV2 raw](nearfield/RGB_DEPTH_BACKBONE_COMPARE_DEV_20261010.md)中远W440/554、F25/199；远带并非全面更好。

【开发观察】[同FREE排序](nearfield/RGB_MATCHED_FREE_ARKIT_CAL_DEV_20261010.md)Uni近带较好；【未确认/负结果】两批各六新visit：[替换](nearfield/RGB_BACKBONE_CONFIRM_DEV_20261010.md)近W3→103/292、F0→12/515；[混合](nearfield/RGB_BAND_HYBRID_DEV_20261010.md)W1→48/159、F1→29/694＝4.18%>2%。双模型本机热态中位0.271秒/帧。

【开发观察】先修[读出尺度](nearfield/RGB_NEAR_READOUT_SCALE_DEV_20261010.md)：全带cut可能超近带余量上限；改尺度又损中远，深度误差未排除。

## 4. 融合：query级，真实RGB＋半合成ToF

【预注册确认】[v2](nearfield/SYNC_FUSION_CONFIRM_V2_DEV_20261011.md)中带+385/1923 POS通过。→【开发观察】[审计](nearfield/SYNC_FUSION_EVIDENCE_AUDIT_DEV_20261011.md)见位置先验混入，用户追溯为Claude指令引入；收益非全由先验解释。

→【未确认/负结果】[v3](nearfield/SYNC_FUSION_CONFIRM_V3_DEV_20261011.md)去位置、加门控后净+87/2077 POS，区间跨0；只比见证FAIL，D′/RGB F4/689与67/689。

→【开发观察】[同FREE事后](nearfield/SYNC_FUSION_V3_MATCHED_FREE_POSTHOC_DEV_20261011.md)：2%下D′/RGB/OR W1755/1089/1495，各/2077 POS；损失多弱证据，强RGB仅18/126，强支持保留式仲裁前提不足。

→【预注册确认】[v4](nearfield/SYNC_FUSION_CONFIRM_V4_DEV_20261011.md)冻结D′，判据取自v3事后；新visit [0,5%]归一化pAUC差D′−RGB +31.9pp、D′−OR +20.0pp，两个95%区间下界>0。部署D′/RGB/OR W/F＝2538/48、1810/38、2178/26（POS2873/FREE866）；按用户取舍判断有利，非双向占优。OOF→冻结分数有偏移。

【未确认/负结果】近带区间跨0；[远带](nearfield/FAR_BAND_FREE_DIAGNOSIS_DEV_20261011.md)未见错标共同主因；退化改善仅描述。

## 5. 数据源

【开发观察】[v1](nearfield/SYNC_RGB_TOF_DATASET_V1_DEV_20261011.md) FARO时间稀疏；[v1.1](nearfield/SYNC_RGB_TOF_DATASET_V1_1_DEV_20261011.md)静态sub-ray平均覆盖66.47%，合格103/384帧，完整源失败；native仍半循环。

复用[v1 manifest](../../../artifacts.local/work/sync-rgb-tof-dataset-v1-dev-20261011/dataset_manifest_v1.json)/[v1.1 manifest](../../../artifacts.local/work/sync-rgb-tof-dataset-v1-1-dev-20261011/dataset_manifest_v1_1.json)；capture/窗口/派生臂随visit隔离，噪声副本不增样本；旧eval已消费。

## 6. 开放问题：按影响排序

【未确认/负结果】①sim→real：台架越线属个人观察；[旧H3](../hardware-bringup/CNH_H3_LEARNED_DEMO_20260928.md)恢复背景左/中HEAD均128/128帧触发。背景扣除、噪声底/查询语义未对齐；非假警率或当前S成绩。②远带FREE读出/标签。③ToF退化。④告警融合缺事件数据，query W非及时检出。⑤细杆。

## 7. 下一步

已定：先用已有真实录制诊断并修复sim→real输入与模拟差距，低成本查背景/串扰、噪声底、坐标和时钟。

备选：远带缓存诊断，分清读出/参考，成本低到中；退化新visit确认，检验坏ToF影响，成本中到高。

## 8. 方法学教训

【开发观察】审先验；同FREE比见证，免低估保守法；开发收益确认时缩水；二元成本门不替取舍；cal族/visit少会漂移，池化仅缓解。
