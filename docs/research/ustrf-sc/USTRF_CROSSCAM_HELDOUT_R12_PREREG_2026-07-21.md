# USTRF 跨相机 marker held-out R1.2 预注册（2026-07-21）

状态：**taxonomy 预检通过，六来源预注册已冻结，尚未解封结果**。本轮不训练、不调 detector/IoU/polygon 阈值，不授权 Android、App 或生产换模。

## 先冻结 detector taxonomy

- 候选：YOLOE-11s segmentation prompted 模型，静态类冻结为 `traffic cone / delineator / bollard`，`imgsz=640`、`conf=.05`、target IoU `.30`。
- 预检仅使用已降级为诊断集的 R1.1 帧，不读取任何 R1.2 新来源。9 个可见目标帧中匹配 6 帧，三种冻结标签均至少出现一次有效目标匹配，故 `taxonomy_contract_passed=true`。
- 这只证明类别清单与非零识别能力，不是召回门：R1.1 的 London 0/2、Cape Town 0/1 仍漏匹配。
- 候选目前只获准作为 offline PyTorch 研究候选；静态 export、Android TFLite parser 兼容性、同设备 benchmark 均未完成，默认模型替换继续关闭。

Detector contract SHA-256：`87ebdf05538d8ccfa1f97037c244820f61c2517c28a979699171f56c295399ed`。Taxonomy audit SHA-256：`3e3da4d5a44d7ec8e70a053caec213ec0294b2dd2943a56bf84a9dcf47b082bc`。

## 新六来源冻结清单

| 关系 | Pexels ID | 冻结目标 | 冻结窗口 |
| --- | --- | --- | --- |
| inside | `4019448` | Thailand/Malaysia 边境通道最近大型锥桶 | 4000–8000ms |
| inside | `5319339` | 桥入口中央灰色伸缩 bollard | 0–5000ms |
| inside | `37839199` | 城市道路施工带最近中央锥桶 | 2000–12000ms |
| outside | `31980662` | 东京作业车左后方沥青锥桶；路线为右侧砖面人行道 | 0–6000ms |
| outside | `35845191` | 封闭标牌后方左侧道路/草边锥桶 | 0–8000ms |
| outside | `10339806` | Vancouver 工区右侧最近柔性标杆 | 8000–16000ms |

六个 provider/source/event ID 均唯一，视频文件与 SHA-256 已绑定；旧 R1.1 六来源没有复用。预注册 SHA-256：`25ac03c7f6a1ef48b671d8d9a57a8316b894ae318b9c018160b1d98dc4233ef3`。

## 解封顺序与硬门

1. 先在旧 R1.1 诊断材料或独立非 R1.2 canary 上完成静态 class export 与 Android parser 收据；R1.2 六来源不得用于调 export/parser。
2. 冻结六来源唯一目标 bbox/contact 与逐帧 polygon 或预声明稳定短窗，先运行 oracle 几何。
3. oracle 完成后才运行冻结 detector；Android 还必须等待 export/parser 与同设备合同闭合。
4. 结果一旦打开不得替换来源、拟合 `.05/.30` 阈值或搜索 polygon；六个来源包括失败都必须报告。

机器校验：`USTRF_R12_PREREG_OK 6 3 3 bollard,delineator,traffic cone`。本轮仍是视觉代理研究证据，不含 human event truth 或设备米制几何。

外部合同依据：Ultralytics 官方 [YOLOE 文档](https://docs.ultralytics.com/models/yoloe/)（prompt classes 与静态导出边界）；Pexels 官方 [License](https://www.pexels.com/license/)（本地研究素材使用边界）。
