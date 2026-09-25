# R1 Street 数据恢复与机械续跑

冻结提交`5c253967`之后，主集唯一运行完成全部预声明臂、train-only K选择、门槛及敏感性，保存`result.json`状态`ALLEY_COMPLETE`。Street入口因副机源尚未传回、指定恢复目录缺raw-manifest.json而停止，`failure.json`保留。该失败发生在任何Street评分之前，不改主集门槛或结果。

允许只恢复原Street960帧已存在源文件（逐哈希验证，不采集）并使用`cnh_r1_readout_street_resume.py`接续1920帧描述性评估。脚本读取已保存的alley train偏置及每臂阈值，校验原冻结代码/协议hash，不重新计算alley、不重拟合/选K、不修改读出定义；Street分数和独立回执另存`street-result.json`，保留原`result.json`及失败。完成后`completion.json`链接两份回执。剩余科学计算预算为1800秒减主阶段已消耗时间，传输等待不算新增科学运行。

传输实现故障可换兼容SSH实现或远端原源执行同代码，但不得重复计算以挑选较好结果、替换源或缩减1920分母。所有实际传输尝试/哈希/资源释放在恢复目录回执记录。此文与续跑代码先提交，再读取新Street结果。

大包的三种SSH实现尝试吞吐均不足后，采用原源所在地各960帧执行：主机`--part main`、副机同冻结arms及alley偏置，只回分数。主机合并NPZ时逐帧核对完整1920的frame_key/labels与原物化数据，以原alley阈值生成统一表；不把两半的AP平均。两端均必须960帧原H3逐位一致，副机回执绑定代码与输入hash。合并是确定性报告，不新增科学臂。
