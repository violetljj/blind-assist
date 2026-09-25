# R1 Street worker执行位置机械恢复

原冻结科学实现提交 `5c253967`，Alley已消费结果不重跑。原完整Street1920由主机960和worker960组成。worker原深度在远端齐备，但本地薄包仅含8帧transport。443,402,680字节源归档已绑定原manifest哈希，系统SFTP首块停滞、系统legacy SCP约2KB/s、Git OpenSSH legacy约15KB/s的机械尝试均不适合完整恢复；原始源保持不动，失败回执保留于本地 `cnh-r1-street-worker-source-20260925-v1`。

经root明确授权，仅将worker960的固定Street评分移到原源所在worker执行。执行前提交 `cnh_r1_worker_street.py`，把本地主机原冻结 `cnh_r1_readout_run.arms/synthesize/evaluate` 及传递依赖按SHA打包；不使用worker旧checkout的科学实现。包内携带已保存 `train-bias.npy`、Alley `result.json` 的22臂原阈值和原worker materialized manifest/H3/targets，全部绑定文件SHA；不重新拟合偏置、选择K、选择阈值、运行Alley、改seed/参数或增加臂。

远端仅访问原 `cnh-street-development-worker-batch-20260924-v1/capture`，必须匹配冻结raw-manifest SHA `25ced0da709c149cd0d78db3fbe50115aa5a8165fca8ebf68c69e8594b97f758`。逐帧验证EXR/mask/camera的原materialized哈希和frame_identity/seed，960帧原H3端点逐位一致；核查24clip×40帧的相机前进向量为[0,0,+0.1]m及固定旋转后，才执行22个原读出臂。每clip独立，不跨主机/布局/片段组合时序。

只返回960×6×22分数、原labels/frame_keys及完整性/固定阈值指标回执；本地主机验证返回标签、frame_keys与原targets一致后，按原主机960顺序拼接为1920。只有合并完整1920才报告Street整体结果，不能用一半冒充完整队列。该执行位置变化不增加科学预算：每分片最多1800秒作为原冻结单次执行的防挂起上限，不是新的参数搜索或重复择优权限。

保留输入绑定、分数、结果、失败和恢复日志；释放任务传输连接、临时源码包与源大ZIP。原worker采集源及既有materialized证据始终保留。无City、test或新采集。
