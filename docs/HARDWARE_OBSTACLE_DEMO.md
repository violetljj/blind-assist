# 8×8 ToF 硬件避障展示

2026-09-22，Android v10.11.0（39）。这是最新主 App 内的独立展示入口。

## 展示方法

选用基础 **ToF 完整区域支持与前方走廊相交**，不选 A*、学习增强或
校准分数筛选。来源是 `ba_camera_corridor.py::tof_readout`，无需模型权重、
RGB 语义类别或 CNH。原始 nominal-45° 模拟基线在已消费 96 帧中为
30 TP / 19 FP / 6 FN、5/6 事件；后续校准版本为 30/15/6，因此这里明确
选择较基础、有误报代价的版本。详见 [原始结果](../research/active/dtr-r0/nearfield/TOF_FOV45_20260920.md)。

硬件适配保留完整区域角范围，以径向测距区间换算轴向范围，再判定是否与
X±0.30 m、Y[-0.20,0.90] m、Z[0.30,3.00] m 相交。45°方形视场、
`0.13 + 0.06 × range` 距离半径和当前装配粗朝向都是展示工程假设，
没有完成实物标定；上面的仿真成绩不属于此硬件版本。它不规划绕行路线。

仅 status=5、有目标、quality=KNOWN 且距离>3 mm 的区域可贡献提示；
某一区无效不会覆盖其他独立有效区域。1–3 mm 是已有异常的排除规则，
不是实物精度保证。缺失、畸形网格、停止采集、断连、过期均为 UNKNOWN；
未提示也不代表可通行。极近处和视场外不在此走廊定义的覆盖承诺内。

## 使用

1. Atom 相机和 XIAO ToF 接电脑；手机通过 USB 连接同一电脑并开启 ADB。
   XIAO 使用 `xiao-tof-8x8-v1`（64 区），CNH 暂不使用。
2. 保持已有硬件观测台在电脑 `127.0.0.1:8766` 运行；若未运行，在仓库执行
   `pwsh -NoProfile -File research/active/hardware-bringup/dashboard.ps1`。
3. 安装本次 APK；执行
   `pwsh -NoProfile -File research/active/hardware-bringup/app-demo.ps1 -Action Start`。
   脚本建立 USB 转发并开启有时限采集，显示实际串口和停止方式。
4. App 首页 → **眼镜设备** → **硬件避障展示（USB中转）**。
   展示页显示相机、原始 8×8 区域、有效数、支持区域和前方提示。
   红色是可能相交支持，中性格也可能无效，不能把颜色理解为安全地图。
5. 在固定朝向下，把较大障碍移入/移出前方视场，观察读数与提示。
   语音开关只影响实时提示；回放始终静音。相机图不叠加假想像素级 ToF 定位。
6. 结束执行
   `pwsh -NoProfile -File research/active/hardware-bringup/app-demo.ps1 -Action Stop`。
   它只清理其回执拥有的录制和 USB 映射，不关闭用户观测台。

数据按电脑接收时间配对，不是曝光同步。实时必须同时具有正在录制、相机与
ToF 新鲜、序号继续推进以及手机请求及时完成；1.5 秒是展示失效上限。
手机前后台切换会停止轮询/语音并清空旧状态。回放有显眼标记，不会冒充实时。

## 验证与产物

纯 Kotlin 几何测试覆盖完整区域、径向/轴向转换、8×8、质量拒绝及 UNKNOWN。
Android 客户端测试覆盖实时/回放、停止录制、序号冻结、慢请求和畸形网格。
实际完成：8 项几何单测、4 项三星 S24 Ultra Android 客户端测试通过；
debug APK、androidTest APK 构建通过，lint 为 0 errors / 18 warnings
（依赖、既有 MainActivity 固定方向、QNN 权限及未用资源，未发现本次新页面 lint 错误）。
APK 包名/39/10.11.0、签名和 16 KB 静态对齐校验通过，已原位升级手机，未清数据。

180 秒有界采集保存 4242 JPEG、919 ToF 帧，双路无解析错误或序号缺帧。
手机截图确认 8×8 显示，相机/ToF 帧号由 2659/3533 增至 3214/3669；
断开转发后旧画面和提示清除，恢复后实时重新显示；回放显眼标记且禁用语音。
中文 TTS 引擎初始化成功；没有录音或用户听感验收，不声称实测可听音量。
动态界面的 accessibility XML 抓取两次未成功，截图可用；不将截图当成正式 UI 自动化断言。

证据：`artifacts.local/hardware-bringup/demo-8x8-20260922/`；
双路原始记录：`artifacts.local/hardware-bringup/paired/dashboard-20260922T100458Z-99cb51/`。
APK：`apk/BlindAssist-v10.11.0-debug-20260922-180828.apk`（相对证据目录），
SHA256 `79DE5B2869D95FB9C0E8B08286F799A97DF1AA215B394CC8A05F729676C2C0E2`。
采集、手机轮询及任务转发已停止；用户原有观测台保留在此次记录回放状态。
单元测试、安装和实机显示只证明对应工程链路，不给出实物准确率或安全效果。
