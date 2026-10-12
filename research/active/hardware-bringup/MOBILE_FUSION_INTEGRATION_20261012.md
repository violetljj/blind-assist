# 手机融合实验接入

2026-10-12，用户授权“把能做的做了”。本轮为工程接入，不训练、不选阈值、不打开新录制评价；原冻结 v1/v2 协议保持原样。

目标：将既有 DAV2-Small、Atom 标定、ToF 强档及 RGB 盒子读出接到独立手机实验入口，验证移植数值、实际推理和生命周期；准备新头戴录制的完整性检查，刷新滞后状态页。

资源：复用已有模型/SDK和已消费单帧，单帧模型设备检查3次；实时工程检查每次至多60秒，只有观察到具体失败才追加。无训练、下载数据集或付费资源。原始产物和失败记录写入 `artifacts.local/work/mobile-fusion-integration-20261012/`。

实时与冻结离线的差别：实时只使用因果接收样本和当帧IMU；冻结ToF使用整段平均姿态，RGB最近帧可在未来到达。手机地面拟合/RANSAC数值实现单列为工程候选，不把端侧跑通当成冻结确认或真实障碍检出成绩。默认辅助不自动替换。

## 交付与使用（2026-10-13）

状态：58 / 10.19.0 已覆盖安装 SM-S9280，保留原录制和设置；工程实验已交付，佩戴后的完整实时 RGB 融合尚待验证。入口为“数据采集 → 打开融合实验 → 开始实验”，低头看空地可标定安装高度，停止/离页关闭资源；当前静音。模型包逐文件 SHA256 核对一致，保留在 App `no_backup/fusion-experiment`，便于后续直接使用。

- `core:vision` 60 项 JVM 检查通过，包括已消费旧帧几何、三姿态投影和高度标定合成检查。冻结输入检查器 2 项合成检查通过；只执行冻结脚本 `--help`，没有新 eval。
- 4 项 Android 数值检查通过：DAV2 三次 DSP 输出与已有 HTP 参考逐值一致；模型 execute（含张量拷贝）155.996 / 121.100 / 123.068 ms，初始化11.397 s，不含预处理/锚定/提示。ToF 三姿态×8帧最大分数误差1.073e-6，热态处理约24–27 ms。图像三次插值/align-corners 与 OpenCV定点去畸变参考通过。
- 首轮真实流8帧全部姿态超范围，严格要求RGB的检查失败，保留 `device-live.log` 和 `live-first-pose-gated.json`。第二次专门检查真实门控：8/8帧ready=false、HEAD/BODY均false，退出后工作线程释放；记录 `POSE_GATED_RGB_NOT_EXERCISED`，未降低生产门控或伪造姿态。165–180 ms是该门控分支的接收到结果年龄，不能称RGB融合延迟。截图人工检查通过。当前硬件摆放下RGB分支没有执行，需佩戴后验证。
- 默认辅助入口1项设备检查通过。debug/androidTest构建及lint通过（0 errors / 88 warnings）；APK版本/签名/16KB静态校验通过。最后只重编译测试APK以记录真实门控分支，不改变产品APK。
- 初版SDK Java AAR的 `libsnpe-android.so` 仅4KB对齐，APK校验拒绝且未安装。已改自建SNPE C API JNI（16KB）及SDK v75依赖；失败记录保留。默认不含SDK的环境仍可构建，实验运行器不可用会提示，不能当作模型已部署。
- APK归档：`artifacts.local/work/mobile-fusion-integration-20261012/apk/BlindAssist-v10.19.0-debug-20261013-001000.apk`，614726656 bytes，SHA256 `BABF5AD957EFB3E02AEC818051F8126C3CA6DD34EFD6DD12000CB10396945F06`。

## 构建与部署

本机QAIRT 2.47.0.260601：`E:/codex-tools/qairt/2.47.0.260601`，可用 `QAIRT_ROOT` 或Gradle `qairtRoot`指定；`enableDav2Snpe=false`关闭可选运行器。当前native桥针对已验证8 Gen3/v75，不承诺其他芯片。使用NDK27、CMake3.22.1。模型来源/导出receipt留在上述任务目录；资产安装命令 `pwsh -NoProfile -File scripts/install_mobile_fusion_assets.ps1`，该脚本只部署已有五文件，不下载或安装APK。

本次Maven连接停滞后，从官方仓库下载ORT1.26.0 AAR并核对官方SHA1；临时本地Maven位于任务目录，构建使用 `ORG_GRADLE_PROJECT_localMavenRepo`，未改依赖版本。新app/.cxx缓存已移入同一artifact任务目录并以junction保持原路径，哈希清单见 `native-cache-migration.json`；hygiene通过。产物实际位于F盘canonical junction。

所有主机构建/设备测试已结束，手机回到首页，无实验工作线程。保留模型包、测试参考及日志供复现，不删除原始采集。测试参考可独立移除；日常实验需保留DLC、ONNX、camera.json、atom_remap.bin和cnh_strong_calibration.bin。

## 尚需真实动作

佩戴当前固定安装，先低头空地标定，再观察有效姿态下RGB尺度来源与融合显示；按冻结协议录制新段并导出。标签仅看RGB并先提交，再执行v1/v2及完整性检查，见[执行入口](FROZEN_FUSION_EVAL_RUNBOOK.md)。原阈值/模型/协议未改；实时因果匹配、Java RANSAC随机序列和在线TLS高度标定与冻结离线实现的差异保持工程候选身份。
