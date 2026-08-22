# Last-10m current-frame visual servo S0v11 result

状态：`COMPLETE_AND_SEALED / CURRENT_FRAME_VISUAL_SERVO_FALSE_COMPLETION / BBOX_HEIGHT_COMPLETION_REJECTED / NO_P1 / DEFAULT_APP_UNCHANGED`

## Answer first

将 semantic proposal 和 leftmost relation selection 接成闭环后，候选覆盖不再是主瓶颈：13 个可评估 case 全部在
bounded pool 中出现过正确 target。但当前 controller 的 bbox-height arrival cue 产生 `9/13` 错误完成，正式结果否决
该 completion interface，而不是把 `1/13` 真完成包装成成功。

## Frozen successor

S0v11 使用 FacadeElements 的下一组未消费 path-hash roster（`skip=48, take=24`）。Goal Contract、source lock、
五类 door taxonomy、view/action graph、controller、两帧确认规则、private completion rule 与 12-observation budget 都在
selected RGB/label access 前冻结。private truth 得到 `13 VISIBLE / 11 NOT_VISIBLE`，达到预注册的 12-case 门后才授权。

环境只对真实 facade RGB 做 deterministic pan/zoom 和第二帧轻微垂直复扫；每次动作后创建新图并重新运行固定
`YOLOE-26n-seg / Ultralytics 8.4.52 / imgsz=640 / conf=0.001 / K=10`。controller 只读取当前候选：无候选时按
leftmost goal 向左扫描，选择 x-center 最左 proposal，然后执行 `TURN_LEFT / TURN_RIGHT / FORWARD /
ARRIVAL_CONFIRM`。状态不保存 candidate id、bbox、feature 或 identity。

private evaluator 才将 selected proposal 与人工 target bbox 比较。true completion 要求连续两个 fresh observation 都命中
同一 private target，并且 target 至少 80% 可见、中央 ray 落在 target 内、normalized target height 至少 0.55。

## Result

| Metric | Result |
|---|---:|
| evaluable visible cases | 13 |
| provider calls | 74 |
| target available in bounded pool at least once | 13/13 |
| target selected at least once | 11/13 |
| true completion | 1/13 |
| false completion | 9/13 |
| no-target-candidate failures | 0 |
| target-not-selected failures | 2 |
| control/completion failures | 10 |
| terminal | `CURRENT_FRAME_VISUAL_SERVO_FALSE_COMPLETION` |

只读 anatomy 显示多数完成由几乎占满 rendered frame 的 YOLOE region box 触发；这些 box 与 door target 可以达到
proposal IoU，却不能证明完整 doorway 仍在视野，更不能证明物理距离足够近。9 个 false completion 中多个 private
target visible fraction 只有约 `0.41–0.75`。因此 proposal extent 与 nearness/completion 是两个不同接口。

## Algorithm decision

下一步不再调 bbox-height threshold，也不在已消费 S0v11 上补跑 mask/model。Brain 必须拆成：

1. semantic proposal 和 relation selection 只负责 target availability/alignment；
2. completion 必须等待独立 nearness/depth + doorway visibility/geometry evidence；
3. 独立 signal 不可用时输出 `COMPLETION_PENDING/ABSTAIN`，不能从 detector bbox 推断已到达。

自动 source discovery 已定位 NYU Depth V2：官方 labeled set 含 1,449 对 aligned RGB、meter-valued depth、dense class
labels 与 instance maps，适合建立 door-region nearness observability cohort。它仍是室内静态 RGB-D，不是最后十米物理
walk-through；下一实验只允许回答独立 depth 是否能阻止 premature completion，不得升级为导航成功。

## Evidence identity

- manifest SHA-256: `54091a58c94fcf69da81eae69de4834be99e2279cb0b253a07969a13450ec8ab`
- roster SHA-256: `735218cf64d59f10e2ba3ab4bd7c06c8fe09595c1541c7019cce5e8e6b86de4e`
- authorization SHA-256: `2c0cd90bd7d8004d2383f37e324705e0e4ab4d2a0b03a5ec73d26072e18da58a`
- run SHA-256: `cd4527097f159a5c5381c1684c977c3f09ea26c611e5bb7e7ed22edbded0c250`
- journal SHA-256: `db3ed72081e2e005ef794834a444312a61d0bc3341859e869a5beb53d5b5676a`
- evaluation SHA-256: `a023bd2567adc8ee1a62873c0739f23956b1aa7370079ad237ca51b542da50e1`

Ignored evidence root:
`artifacts.local/evidence/last-10m-visual-servo-v1/s0v11-fresh-facade-v1/`.

## Claim ceiling

本结果只支持 real-facade/synthetic-view current-frame control failure anatomy。它不证明真实转头、物理接近、door
traversability、建筑归属、盲人可用性、产品或安全有效性；默认 App 未改变。
