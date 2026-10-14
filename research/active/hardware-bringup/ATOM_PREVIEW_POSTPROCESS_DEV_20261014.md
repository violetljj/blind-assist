# Atom 采集预览去畸变与轻增强

2026-10-14 · EXPLORE · 用户授权“可以试试”。范围为采集页显示处理、离线画面对照与可安装试用包；不运行新冻结评价，不改变模型、原始记录或相机固件。用户未设置计算预算，本轮只运行相关像素检查、构建和 APK 检查。

## 实现

- 采集预览新增独立“去畸变”“轻增强”开关，默认开启并记住选择。两者关闭即显示原始画面；保留原有录制、JPEG 字节与导出链路。
- 去畸变复用 2026-10-12 标定的固定 OpenCV remap，SHA256 `97d14a4ac5e8be48ceb16cbd1101eb7387e8cc8c10a4fef558d3139c27f07c63`。从 App 私有 `no_backup/fusion-experiment/atom_remap.bin` 读取；缺失、身份损坏或画面不是 640×480 时保留原始视场并显示提示。已校正图为 480×640 直立画面，不再次旋转或镜像。
- 轻增强为固定 256 项 LUT：`x=(v/255)^0.96; y=x+0.60*x*(1-x)*(2*x-1)`，输出四舍五入至 0–255，保留黑白端点。无逐帧自适应、锐化或生成式细节。
- 处理运行在已有单线程 latest-only 预览 worker；切换递增代次，重新处理最近有效帧，离页后不发布。复用原始像素读取缓冲；校正输出数组和 Bitmap 按帧分配。
- 性能收据保留 JPEG `decode_ms`，新增独立 `preview_processing_ms`（含像素读取、处理及 Bitmap 创建）。这些不是曝光到显示延迟。

## 验证与交付

- JVM 像素检查 **6/6 通过，0 skipped**：四种开关组合、原像素保留、独立 OpenCV 去畸变/直立参考逐像素相等、曲线端点与 alpha、缺失/尺寸不符 fallback、损坏 map 拒绝。参考覆盖 307200 像素。
- `:app:assembleDebug`、`:app:assembleDebugAndroidTest` 和 `:app:lintDebug` 通过；instrumented APK 仅构建，未在设备运行。APK package/version/debug 签名与 16 KB 静态对齐检查通过。
- 两张旧 Development 实拍帧 H4/H5 的对照已目视检查：几何弯曲减弱，层次略改善，原有模糊仍存在。Python/OpenCV 主机处理约 1.3–1.4 ms 去畸变、0.2 ms 增强（各 20 次中位数）；这些不是 Android 实测。
- 试用包 **62 / 10.19.4**，498121312 bytes，SHA256 `e82a5b8df102fcd962dff93ee4f68b2350fb2b3fa49696283c00c19242e7cd86`，归档为 `apks/BlindAssist-v10.19.4-debug-20261014-114635.apk`。

对照图 `comparison-1.png` / `comparison-2.png`、输入/标定 hash、脚本、Gradle 日志和 `validation.json` 保留于 `artifacts.local/work/atom-preview-postprocess-20261014/`（物理 F:）。本轮设备健康检查没有在线手机或已有 AVD，**未覆盖安装，未验证真机观感、FPS、持续运行、TalkBack 或大字体交互**。去畸变会裁剪视场；显示增强的收益不能直接解释为模型检出收益。
