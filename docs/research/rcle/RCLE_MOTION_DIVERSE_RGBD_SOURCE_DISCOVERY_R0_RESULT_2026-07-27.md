# RCLE motion-diverse RGB-D source Discovery R0

日期：2026-07-27

## 结论

本轮只做 metadata/source-access 筛选，最多保留三个候选。终态为：

`SELECT_ETH3D_DESK_3_FOR_GEOMETRY_ONLY / METADATA_VALID_ROLE_UNKNOWN`

排名只表示“以最低成本证伪 geometry admission”的顺序，不授予任何运动角色。
尤其是 trajectory、序列名和自然语言描述都不能直接产生
`POSITIVE_APPROACH_WINDOW` 或 `BELOW_TRIGGER_REFERENCE_WINDOW`。

| 排名 | 候选 | metadata 优点 | 缺口与处置 |
| ---: | --- | --- | --- |
| 1 | ETH3D SLAM `desk_3` | 2061 帧，约 76 秒；同步 global-shutter RGB-D；训练集 timestamped camera-to-world ground truth；mono/depth 分包；两个 ZIP 均支持 Range | 官方未发布 cryptographic checksum，明确记为 `UNKNOWN`；动作描述不能授予角色。只允许先取 ZIP central directory 与 `groundtruth.txt`，再冻结四个 10 秒窗并按需取 depth member |
| 2 | OpenLORIS corridor family | 真实 D435i aligned depth、timestamp、GT；独立 11 MB groundtruth；HF LFS 给出 exact bytes 和 SHA-256 | depth 位于 13.85/19.70 GB package，current metadata 尚未闭合精确 member offset；不进入本轮 payload |
| 3 | ICL-NUIM `lr kt3` | 1242 帧、30 Hz、42 秒；单独 TUM-format pose；depth/pose ground truth；CC BY 3.0 | synthetic；仅勉强容纳四个非重叠 10 秒窗；官方 checksum 未发布；不进入本轮 payload |

CID-SIMS Floor3 全部退出当前漏斗，`floor3_3` 明确禁止。ETH3D
`sofa_3`、EVIMO2 `sanity_ll` 和 Bonn 已 burned，不作为新候选。CoRBS 当前
官方入口/bytes/checksum 不闭合；TUM Kinect RGB/depth 非硬同步且 payload
捆绑；它们不进入前三。

## 第一名的冻结边界

唯一允许进入 geometry-only 的 identity 是 ETH3D `desk_3`：

- mono ZIP：
  `https://www.eth3d.net/data/slam/datasets/desk_3_mono.zip`，
  `788957601` bytes，ETag `5cf57ceb-2f0689a1`；
- RGB-D ZIP：
  `https://www.eth3d.net/data/slam/datasets/desk_3_rgbd.zip`，
  `220689904` bytes，ETag `5cf57e22-d2775f0`；
- 许可：`CC BY-NC-SA 4.0`；
- 官方 cryptographic checksum：`NOT_PUBLISHED_OR_NOT_VERIFIED`。

ETag、Content-Length、ZIP CRC32 和本地 member SHA-256 只能绑定本次传输，
不能伪装成官方 cryptographic checksum。若 HTTP Range、central directory、
timestamped pose、depth member selective extraction 或同步链任一不闭合，
立即 `NOT_EVALUABLE`；禁止退化为完整 mono/RGB 下载、换候选或补窗。

## 冻结漏斗

1. 在任何第一名 payload GET 前完成 burned Floor3_2 W3 smoke。
2. 通过稳定 root adapter `scripts/fetch_remote_zip_members.py` 取得 central
   directory 和最小 pose member；外部合同不引用研究内部 transport 脚本。
3. trajectory 只用于按预冻结、outcome-blind 规则确定四个 10 秒候选窗身份；
   它不授予角色。
4. 只取得四窗所需的 depth members，使用冻结 geometry 公式判定
   `2 positive + 2 below-reference`。
5. 不足即 `NOT_EVALUABLE` 且 RGB bytes 保持 0；满足才冻结四窗身份并另立
   RGB 运行。

## 官方来源

- [ETH3D SLAM datasets](https://www.eth3d.net/slam_datasets)
- [ETH3D SLAM format](https://www.eth3d.net/slam_documentation)
- [ETH3D license](https://www.eth3d.net/)
- [OpenLORIS-Scene source and sensor authority](https://lifelong-robotic-vision.github.io/dataset/scene.html)
- [OpenLORIS maintained download index](https://huggingface.co/datasets/shixuesong/openloris-scene)
- [ICL-NUIM dataset](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)
