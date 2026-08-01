# HFTF Stage C D2 六源短路径媒体获取结果

## 结论

唯一一次正式获取成功，终态为
`D2_SIX_SOURCE_SHORT_PATH_MEDIA_COHORT_ACQUIRED`。合同锁定的 6 个
official-train Development parents 均已物化；每个 parent 含 13 个 normalized
5 Hz frames 的 RGB、panoptic mask、metric depth，以及 13 个独立 pose slices。

这只建立 synthetic Development 媒体 transport、完整性与 future-blind 输入接口，
不建立 geometry teacher、模型 effect、人类事件真值、安全性或主线晋级证据。

## 唯一一次执行

execution contract、acquirer、test 与 SANPO network transport dependency 由 commit
`1f04af5bb77acee45ce3432c5d5ce0d5784f8c92` 提交推送，正式运行前再次确认
`HEAD == origin/master`。CLI 只启动一次；exclusive attempt 在首网前完成
`flush + fsync`，原进程自然退出。

254 个下载请求全部在 attempt 1 成功，retry line 为 0，stderr 为 0 bytes。6/6
source 完成后才把整个 staging 原子发布为 cohort；没有 failure terminal、重跑、
换源、追加或 partial fill。

## 离线完整性复算

独立离线复算得到：

- 378 个 artifact files，共 300,811,962 bytes；
- 234 个媒体对象，共 299,513,891 bytes；
- 6 个 pose CSV，共 47,240 bytes；78 个 pose slices，共 42,057 bytes；
- 6 个 source transport receipts；
- 所有 frozen generation、size、MD5 与本地 SHA-256 绑定一致；
- 所有 pose slices 与 selected source frame、READY state、finite position、
  unit xyzw quaternion、source pose CSV SHA 一致；
- final 最大路径长度 168，低于 240；无 `.tmp`、staging 或 failure 残留。

审计只为 size/MD5/SHA 读取媒体 bytes，没有解码或视觉检查 RGB/mask/depth；pose
slices 只用于 schema、index、finite、quaternion 与 hash 校验。future truth、
geometry teacher、candidate effect 均未打开。

关键 evidence SHA-256：

- attempt：
  `8153156da811807e927c600ce12342b640eee8ae8f481587f4b08cc292cc3117`
- cohort manifest：
  `07b968e97c1a010c7d49beff6d09dc2fb8677826680be6ea4efc235aedd355c4`
- acquisition receipt：
  `59c9677393b06809b160163b81c918c6635c0fe6db2e6c12ba13b027e39667a6`
- per-frame acquisition index：
  `60e63e2df8b2813519e90a287b841dbcfa2b2c9a9b0765b1f10ebcf7c9c8b2a8`
- offline validation：
  `62abd95c32926417b04986b1872c45951a64a307cb74f0549ac1f0f43ac186c4`

## 后继权限

本结果只授权冻结另一份 hash-bound D2 mechanics execution contract。该合同必须先
提交推送，再按顺序运行 future-blind preprocessor 与 one-shot truth/effect evaluator；
在 42 个 anchor predictions、84 个 horizon records 全部 durable 前不得打开 future
pose/depth/mask，truth join 必须由 exclusive receipt 限制为一次。

RGB student、reserved official-test、研究主线、默认 App、Android、生产与 safety
权限继续关闭。
