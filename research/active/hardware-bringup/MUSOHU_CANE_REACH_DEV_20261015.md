# MuSoHu 适用性检查：户外盲杖漏检物来源（2026-10-15，EXPLORE，停止）

MuSoHu（GMU，CC0）：头盔 VLP-16 激光雷达 + ZED2 + IMU，约 20 小时行走。下载 `07032023_PR_GMUCampus2_4_neutral.bag`（1,707,661,433 B，MD5 8f9ae966b76932005d1d4b831bd8b4dc 一致），存于 `artifacts.local/downloads/musohu-20261015/`。

## 发现

1. **ZED 深度不可用**：`/zed2/zed_node/depth/depth_registered/compressed` 的格式为 `16UC1; jpeg compressed`，即 16 位深度被转成 8 位 JPEG，像素大多饱和在 255，米制深度已丢失。脚本按深度+里程计算出的计数（相机离地 0.83 m 明显不合理）作废。
2. **所选包是室内**：按名称选的 "GMUCampus2_4" 均匀抽 12 帧全部是室内走廊/楼梯；户外包（old town、memorial、outlet、dc_night 等）每个 1.1–2.8 GB。
3. 剩余可行路径只有激光雷达：16 线 ±15°，近距只扫到约 1.4 m 以上，需要按 `/tf_static` 外参和里程计融合前方 4–7 m 时的扫描来覆盖身体带与地面；工作量与下载量都明显更大。
4. 下载过程：24 路并发触发服务器限流（mod_qos），IP 被封约 1 小时；改 3 路后完成（403 s）。以后对该服务器并发不超过 3。

## 判断

按效率止损：MuSoHu 不作为当前盲杖漏检物来源。若以后仍需户外样本，优先考虑激光雷达融合 + 单个户外包的试读，或改为自录少量悬挑场景。

脚本 `host/musohu_cane_reach_dev_20261015.py` 保留（深度路径因数据格式不可用）；成本：下载 1.7 GB、CPU 约 1 分钟。
