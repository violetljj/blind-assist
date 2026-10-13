# Aria Digital Twin（ADT）数据接入

2026-10-13，执行者 Claude。经用户授权，用户选定的邮箱提交了 ADT 访问表单，也就是同意了 [ADT 数据集许可协议](https://www.projectaria.com/datasets/adt/license/)（2023-09-24 版）。

## 为什么要它

ADT 用 Aria 眼镜头戴采集：鱼眼 RGB、IMU，**每帧都有公制深度真值**和 6 自由度头部姿态，场景是公寓和办公室，共 236 段，有人在其中活动。视角和 Atom 头戴相机接近，可以逐帧算出报警盒里到底有没有东西，从而在几千帧上客观评测 RGB 分支的误报和漏报。ToF 需要用真值深度模拟，因为 ADT 没有 ST 8×8 直方图。

## 许可要点（必须遵守）

- 只能用于非商业研究。
- 发表时必须注明：“Aria Digital Twin Dataset, Meta Reality Labs Research”。
- 不得分发或公开展示数据本身及其派生作品。论文可以发表观察、分析和汇总结果，但不要附带原始帧或深度图。
- 对方会记录访问者的邮箱、设备信息和大致位置。

## 已取得

- 链接文件 `ADT_download_urls.json`：签名的 CDN 链接，约 14 天后失效，可以在页面上重新领取。只放在被 git 忽略的 `artifacts.local/downloads/aria-adt-20261013/` 下，不提交。
- 下载工具：`projectaria-tools`，安装在 `E:/codex-tools/tools/venvs/projectaria-tools`，命令为 `aria_dataset_downloader -c ADT_download_urls.json -o data -l <序列> -d 0 6 8`，其中 0 是 main_vrs，6 是 main_groundtruth，8 是 depth。
- 官方样例 `Apartment_release_golden_skeleton_seq100_10s_sample_M1292`，共 824 MB，用 `AriaDigitalTwinDataProvider` 读取正常：300 帧 RGB（9.97 秒），RGB 为 1408×1408 鱼眼图，深度为 uint16 毫米（中位 2.0 m，86% 像素有效），设备姿态有效。

## 容量

完整的一段序列（vrs、真值、深度）平均约 7.4 GB，其中深度约 5.5 GB，包含 3 路相机，没法只下 RGB 那一路的深度。全部 236 段约 1.7 TB。F 盘目前只剩约 16 GB，所以下一步最多再下 2 段；要下更多，需要先清理空间。
