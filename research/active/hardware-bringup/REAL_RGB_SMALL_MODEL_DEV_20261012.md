# 手机可用的小深度模型：在 RGB 或 ToF 融合中的对比

2026-10-12 · EXPLORE · 执行者 Claude（用户授权，含下载）。接续 [RGB 或 ToF 融合](REAL_RGB_TOF_FUSION_DEV_20261012.md)。

## 结论

**小模型能保住大部分细杆能力，但还不及大模型。候选为 DAV2-Small（相对深度）＋ToF/地面锚定，518 px 输入。** 正前方竖杆平视/低头两段 BODY 报警 67% / 92%（UniDepthV2-L 为 72% / 95%），空通道 0–3%，笔记本 GPU 每帧约 56 ms。弱点是头部横杆：小模型 HEAD 只有 9–14%，大模型为 66%，这部分要靠 ToF 强档（43%）补。DAV2-Small 公制版（Hypersim）在大厅里把距离估远，两段竖杆只有 0% / 18%，淘汰。UniDepthV2-S 速度快（31 ms），但细杆只有 33% / 77%。

所有片段都已看过，标签是 Claude 判读的；像素门槛、旋转、检测区与大模型读出完全相同，没有为小模型调整门槛。地面锚定的头高是事先设定的（1.35 m）。

## 对比（大厅＋办公室 18 段，ToF 每隔一帧取一帧）

| RGB 深度 | 参数 | 笔记本 GPU 每帧 | H5 正前杆 BODY | H8 正前杆（低头）BODY | H7 横杆 HEAD | H6 侧杆 BODY | 空通道（H0/H1/H2、电视墙） |
|---|---|---|---|---|---|---|---|
| UniDepthV2-L（参照） | ~3.5 亿 | 未精确计时（~150–200 ms） | 72% | 95% | 66% | 21% | 0% |
| **DAV2-S 相对＋ToF/地面锚定，518 px** | 2480 万 | 56 ms | **67%** | **92%** | 11% | 17% | 0–3% |
| DAV2-S 相对＋锚定，308 px | 2480 万 | 23 ms | 33% | 62% | 9% | 17% | 0–9% |
| UniDepthV2-S（ViT-S，已知内参） | 3420 万 | 31 ms | 33% | 77% | 14% | 4% | 0–3% |
| DAV2-S 公制（Hypersim），518 px | 2480 万 | 57 ms | 0% | 18% | 0% | 0% | 0–12% |

门、柱子和近处家具对所有小模型都能报警；融合（ToF 强档或 RGB）后，门/柱子为 80–100%。

**锚定**：相对深度满足 disparity = a/z + b。ToF 有 ≥6 个有效 zone 时用 ToF 拟合，否则用地面锚定：由 IMU 姿态、相机←ToF 旋转和假设头高，得到画面下方地面像素的公制深度，RANSAC 拟合；两者都失败就沿用本段上一帧。平视开阔场景中，ToF 在 4 m 内常常没有有效读数，这时没有地面锚定，竖杆那段就完全没有 RGB 输出。头高敏感性：设 1.35 m / 1.45 m 时空通道安静；设 1.25 m 时，朝电视墙那段 BODY 误报 57%。所以头高需要实测，或者由 ToF 看到地面时在线估计。

## 权重与许可证

- **DAV2-Small 相对**：官方 `depth-anything/Depth-Anything-V2-Small`（`depth_anything_v2_vits.pth`，99,218,434 B，sha256 `715fade1…1378`），模型卡写明 Apache-2.0。与 DEPTHOR-ZJU-Small 中嵌入的 `depth_anything.*` 239 个参数逐值完全一致。
- **另外下载、仅作对照**：DAV2-Small 公制 Hypersim（99,222,290 B），UniDepthV2 ViT-S（136,777,580 B）。这两份的许可证未核验。

收据在 `artifacts.local/models/*/weight_receipt.json`。

## 手机实测（同日追加）

官方 DAV2-Small 先导出成固定尺寸 ONNX（opset 17，NCHW），再用 QAIRT 2.47 `qairt-converter` 转成 FP32 DLC（2408 万参数）。在 SM-S9280（骁龙 8 Gen 3 / SM8650，HTP v75）上用 `snpe-net-run --perf_profile burst` 跑 21 次推理取平均，输入为一帧真实大厅画面（已去畸变）：

| 输入（高×宽） | HTP（NPU，FP16 在线准备） | GPU（FP16） | CPU |
|---|---|---|---|
| 518×392 | 67 ms | 83 ms | 407 ms |
| 686×518（融合候选） | 120 ms | 190 ms | — |

HTP 首次创建需要 5.2 s / 8.3 s（之后可用离线图准备缓存）。与 PC 上 FP32 PyTorch 输出相比：HTP 相关系数 0.99999、中位相对误差 0.16–0.19%（最大 2.3–3.5%）；GPU 相关系数 0.9998–0.9999、中位相对误差 0.5–0.8%。**686×518 在 HTP 上 120 ms，能跟上 5 Hz ToF，精度无可见损失。** 计时只覆盖网络推理，不含去畸变、缩放、锚定和读出（这些在笔记本上都是毫秒级），也没有测功耗和发热。导出脚本见 [dav2s_mobile_export](host/dav2s_mobile_export_dev_20261012.py)，手机端脚本见 [dav2s_phone_bench.sh](host/dav2s_phone_bench.sh)，日志在 `artifacts.local/work/dav2s-mobile-20261012/phone/`。手机上的临时文件已删除。

## 头高在线估计：不采用

尝试过用 ToF 锚定帧的公制 RGB 点云估地面深度（地面像素高度直方图的众数），结果不可靠：办公室几段估成 0.83–0.89 m，把桌面这类水平面误当地面；门那段也估成 0.83 m。而真正需要地面锚定的平视大厅段，又没有 ToF 锚定帧可用。改为**开机一次性标定**：用户低头看地面几秒，用 ToF 地面拟合得到头高（本批 H0 段 ToF 拟合为 1.373 m，与默认 1.35 m 一致），标定前用 1.35 m。输出见 `rgb_small_model_online_height.json`。

## 下一步

1. 用新的独立头戴录制做冻结检验：标定、旋转、门槛、模型、头高规则都已固定。
2. 接入 App：DAV2-Small 走 QNN HTP（App 已集成 QNN 运行库），加上去畸变与 ToF/地面锚定读出，再加开机低头标定；需要实测端到端延迟、功耗和发热。

脚本：[DAV2-S 相对＋锚定](host/real_rgb_small_model_dev_20261012.py)、[DAV2-S 公制](host/real_rgb_metric_small_dev_20261012.py)。UniDepthV2-S 用融合读出脚本换权重运行，输出在 `rgb-real-feasibility-20261012/unidepth_vits14/`。
