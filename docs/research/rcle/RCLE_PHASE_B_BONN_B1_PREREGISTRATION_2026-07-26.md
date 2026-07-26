# RCLE Phase B Bonn B1 预注册

状态：`DESIGN_REVISION_R5_FROZEN / EXECUTION_NOT_YET_AUTHORIZED`

日期：2026-07-26

唯一机器真源：
`RCLE_PHASE_B_BONN_B1_DESIGN_LOCK_2026-07-26.json`。本文解释该 lock；
implementation 和 validator 必须逐字段消费 machine lock。两者漂移即
`INVALID_EXECUTION_CLOSE_B1`。

## 结论与不可逆阶段

B0 R1 已冻结固定六条 Bonn 序列及十个半开 `10 s` 窗口，但没有授权读取
pose 数值、解码 RGB/depth 或计算 Phase B 指标。B1 严格拆成：

1. `B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION`：只用 source-native pose/depth
   冻结 pair/grid truth 与窗口角色；不得解码 RGB 或计算 RCLE/M2。
2. `B1B_FROZEN_TEN_WINDOW_METRIC_AUDIT`：只消费 immutable、独立验证为
   `VALID` 的 B1A ledger；不得修改 truth、角色、窗口或阈值。

两个阶段不能合跑，各自需要 implementation lock、独立实现审查、唯一
canonical claim、receipt schema 和独立 validator。若 B1A 只有一个条件分支
达到至少两个不同 sequence，B1B 只可运行该分支并保留另一分支完整
`NOT_EVALUABLE` 分母；两分支都不足时 B1B 关闭。任何 B1 终态都不自动授权
Phase C、增强模块、Replay、Android、人体、安全或生产。

## 上游、十窗和 source authority

冻结上游：

- B0 receipt：
  `dc0ffe9a890b539478ff4c035b4dfadea6c21347a11b36f164810a18eb811f86`
- B0 window denominator：
  `f1e6f7f2e54da349d004af744573884e6273089f67bda86d5f0eb812234aa05b`
- cohort identity：
  `513b770d18489fd0caf84874e9fb89456eb3a992fc262b037220b66b5caae86e`
- sequence/window：六条、十窗，rank 分布 `2/1/1/2/1/3`；十个
  `(sequence_id, window_rank, exact Decimal start, exact Decimal end)`
  canonical key 全量写入 machine lock。

十窗必须在任何 ZIP payload read 前预分配。所有 adjacent RGB pair、失败、
缺失和 `NOT_EVALUABLE` 均保留；禁止换窗、删窗、跨窗 pair、延长窗口、
替换 sequence 或把缺失回填为零。window 是主统计单位，sequence 是 cluster；
十窗不是十个 session，六条也不声称六次独立采集。

source authority 固定为：

1. Bonn official page cached bytes：
   `artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_metadata_r0/official_page.html`，
   SHA-256 `2bd8df16acad79c70e1021f1da039c78510034fd9091fd706f8a3f480ea5c186`。
   它给出：Bonn 与 TUM RGB-D 相同格式、depth 已配准到 RGB、RGB
   intrinsics、Brown-Conrady/OpenCV 顺序的 `k1,k2,p1,p2,k3` 数值。
2. TUM official file-format cached bytes：
   `artifacts.local/datasets/egomotion_compensated_looming_r1/bonn_b1_authority/tum_rgbd_file_formats.html`，
   SHA-256 `721c8df093ade2b0078215c3154f6f1a3641a0c691b5123cd037e87b61b30107`。
   它给出：RGB `640x480` 8-bit PNG、depth `640x480` 16-bit monochrome
   PNG、1:1 pre-registration、depth factor `5000`、zero 为 invalid，以及
   `timestamp tx ty tz qx qy qz qw` 中 translation 是 color-camera optical
   center 在 world 中的位置、quaternion 是 optical center 对 world 的方向。
   官方同时建议对 pre-registered 数据使用默认参数、不再 undistort。

因此 B1 固定：

- 图像域：source-provided `640x480` registered pixel domain；
- `K=[[542.822841,0,315.593520],[0,542.576870,237.756098],[0,0,1]]`；
- distortion `[0.039903,-0.099343,-0.000730,-0.000144,0]` 只记录，
  B1A/B1B 都不得再次 undistort/remap；
- depth meters = uint16 / `5000`，only zero invalid；
- pose row 是 `T_world_from_camera`。

若 authority cached bytes/hash、field mapping 或 archive SHA 不符，当前 run
直接 `INVALID_EXECUTION_CLOSE_B1`，不能选其他 calibration。单个 referenced
PNG 或索引引用缺失、损坏、shape/dtype 不符是保留在分母中的 source/method
abstention，不是 execution-contract INVALID。

## B1A：source-native geometry admission

### 唯一执行与读取防火墙

canonical output：
`artifacts.local/evidence/rcle_phase_b_bonn_b1/b1a_geometry_admission`。
目录由审查后的 setup 单独预建，不接受 CLI/env/path override。formal runner
只接受空 argv；validator 只接受 `--validate-existing`。

B1A formal runner 的第一项 application data file operation 必须是
`O_CREAT|O_EXCL` 创建并 fsync `run_claim.json`。claim 前除 hash-bound
bootstrap runner 与 Python runtime/stdlib 外，禁止 project import、B0/lock/
source/ZIP read、pose parse、depth decode、network、stat/glob/listdir。
claim 内嵌并绑定 prereg/design/implementation/bootstrap、B0 receipt/window、
两个 authority snapshot 与六 ZIP hash。success、failure、exception 或
interruption 都消耗唯一 claim并禁止第二次 formal run。receipt/ledger 先写
同目录临时文件、flush+file fsync 后原子替换。POSIX 固定为 `os.replace`
后 fsync directory；Windows 因 Python `os.open(directory)` 不可用，固定为
`MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH)`，不得把无法执行的 directory
fsync 伪装为成功。任一 replace 失败即 `INVALID_EXECUTION_CLOSE_B1`，不得在
replace 后修补或重写；claim 永久
保留。

claim 后 B1A 只允许读取固定 ZIP 中十窗所需的三个文本、八列 pose 数值、
被 `depth.txt` 引用且唯一存在的 depth PNG。B1A 对 RGB 只验证引用路径、
central-directory 中大小写精确唯一 file member 与 declared size `>0`，不读
任何 RGB member bytes，也不声称验证其 PNG header/shape/dtype；这些只在
B1B 首次实际 decode 时验证。禁止 RGB PNG read/decode、RCLE/M2、
旧 Bonn manifest/trace/truth/result、static map、人工/模型画面检查和 network。

### 索引、member 与 pair denominator

- raw bytes 只按 LF 分行；每行 strip ASCII whitespace
  `09/0a/0b/0c/0d/20`。strip 后空行或首 byte `#` 跳过；`data_row_rank`
  只对保留的数据行从零递增。其余行按一个或多个 ASCII whitespace split：
  RGB/depth 精确两 token，groundtruth 精确八 token。numeric token 必须匹配
  ASCII `^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$`；
  timestamp 用 finite Decimal 且严格递增，其余 pose numeric 转 finite
  float64；path token strict UTF-8、无 embedded whitespace。
- 引用必须为规范相对 POSIX path、无 absolute/drive/`..`/NUL，大小写精确且
  ZIP file member 唯一、declared size `>0`；B1A 不验证 RGB header。
- 每窗完整记录所有 chronological adjacent RGB rows，不跨 `[start,end)`。
- `candidate_pair_denominator` 是其中 `dt` 落在闭区间 `[0.020,0.050] s`
  的 pair，在任何 depth/pose join 前冻结。
- dt-outside pair 留在 `all_adjacent_pair_denominator` 和 ledger，但不进入
  truth/method coverage；RGB member、pose、depth 或 geometry 失败仍留在
  candidate denominator。

### depth 与 pose join

depth assignment 的 scope 是每个固定窗，窗内全部 chronological RGB rows
（包括不属于任何 dt-valid candidate pair 的 row）依次处理。对每个 RGB row，
只在同一半开窗中尚未使用且
`abs(depth_timestamp-rgb_timestamp)<=0.020 s` 的 depth rows 中，按
`(absolute_delta, depth_timestamp, source_row_rank)` 最小值选一个；无候选
则 unmatched。禁止窗外 depth 和跨窗竞争。同一个 assigned depth 可合法服务
共享该 RGB endpoint 的前后两个 candidate pair，但不能分配给第二个 RGB row。

pose 允许使用 full sequence 中包围 RGB timestamp 的相邻 rows，即 bracket
row 可落在窗外；只要二者 span `<=0.050 s`，不得外推。精确 timestamp 直接
取该行。translation 线性插值。raw quaternion 必须 finite、norm positive 且
`abs(norm-1)<=0.001`，随后归一化；否则 unmatched。SLERP 固定为：

1. 若 dot `<0`，右 quaternion 取负；
2. clamp dot 到 `[-1,1]`；
3. dot `>0.9995` 时 normalized linear interpolation；
4. 否则用 standard shortest-arc sine-weight SLERP。

quaternion 使用 Hamilton `xyzw` active rotation。令 pose 行为：

```text
T_world_from_camera(t) = [R(q), C_world; 0 0 0 1]
T_current_from_previous =
    inverse(T_world_from_camera(current)) * T_world_from_camera(previous)
R_current_from_previous = T_current_from_previous[:3,:3]
translation_speed = norm(C_world_current-C_world_previous) / dt
angular_rate =
    acos(clamp((trace(R_current_from_previous)-1)/2,-1,1)) / dt
```

angular rate 的单位固定为 radians/s，直接使用 rotation trace，不做
matrix→quaternion 转换。rotation role gate 固定比较
`window_median_angular_rate_rad_s >= 5 * numpy.pi / 180`，不得将
radians/s 直接与数值 `5` 比较。B1B
传给既有补偿核的是：

```text
H_previous_to_current = K * R_current_from_previous * inverse(K)
```

该核内部 inverse warp current 回 previous。90-degree yaw、identity、
`q/-q`、translation-only 与 checkerboard inverse-warp fixture 必须在实现锁前
通过。

### static-surface truth

source depth 由 Pillow `12.2.0` 的 `Image.open(BytesIO(bytes))` 后立即
`load()`；要求 `format=="PNG"`、`size==(640,480)`，`np.asarray` 结果精确为
native `uint16`、shape `(480,640)`、单通道；禁止 convert、rescale、remap 或
色彩/endianness 手工变换，不符则相关 row/pair abstain。对每个 previous grid，
只取全图 raster 中
`x mod 4 == 0 && y mod 4 == 0` 的 deterministic lattice points。grid 归属仅由
previous pixel 决定，并使用与 locked local-affine `_cell_bounds` 完全相同的
`int(round(index*extent/3))` 边界。

有效 previous depth 反投影为 `X_previous`，再用
`T_current_from_previous` 得到 `X_current_predicted`。projection 只需落入
current image 且最近整数中心能容纳完整 `3x3` patch，不要求留在同一 grid。
非负坐标 nearest integer 固定为 `floor(value+0.5)`；patch 不 clip/pad。

z-buffer 在一个 pair 的全部九个 previous-grid lattice points 上全局执行；
跨 previous grid 投到同一 current integer pixel 也只留一个 winner，其 grid
归属仍是 winner 的 previous anchor。先按最小 predicted `z`，再按
`previous_raster_rank=y*640+x` 留唯一 winner。current observed depth
是完整 `3x3` 中全部 nonzero depth 的中位数；无 nonzero 即 missing。
static-consistent 必须：

```text
abs(z_observed-z_predicted) <= max(0.10 m, 0.05*z_predicted)
```

四个计数严格定义：

- `N_previous`：该 previous grid 中 lattice 上 nonzero depth 点；
- `N_projected`：正 `z`、projection/patch 有效且 z-buffer 后的唯一 winners；
- `N_observed`：其中 current patch 有 nonzero median 的 winners；
- `N_static`：其中通过 z consistency 的 winners。

truth-eligible grid 同时要求：

- `N_previous >=30`
- `N_projected/N_previous >=0.50`
- `N_static >=30`
- `N_static/N_projected >=0.50`

同一静态点的 closing truth 使用欧氏 range，而 z 只用于投影一致性：

```text
r_previous = norm(X_previous)
r_current = norm(X_current_predicted)
c_truth_grid =
    median_static(log(r_previous/r_current) / dt)
```

正值表示接近。禁止用 RGB、LK、RCLE/M2、视频观感、sequence 名称语义或旧
结果选择 point/grid/pair/window/role。

B1A 数值环境冻结为 Python `3.11.9`、NumPy `2.1.3`、Pillow `12.2.0`；
geometry/matrix/projection/SLERP 全部使用 IEEE-754 float64，
matrix inverse 用 `numpy.linalg.inv`。除 timestamp/window key 的 Decimal 外，
本文全部 median 均为 `numpy.median(float64)`：偶数个元素时取排序后两个中心
值的 float64 算术平均。

### coverage、嵌套聚合与角色

`truth_covered_pair` 是 candidate pair 中 pose/depth 有效且至少 `5/9`
truth-eligible grids 的 pair。window truth coverage 精确为：
`truth_covered_pair_count / candidate_pair_denominator_count`；denominator
为零或 point `<0.80` 均为
`NOT_EVALUABLE_SOURCE_NATIVE_TRUTH_COVERAGE`。

聚合顺序固定：

- pair truth closing = median eligible-grid `c_truth_grid`；
- window truth closing = median truth-covered-pair truth closing；
- window angular/translation rate = 分别对同一 truth-covered pair 集合取
  pair angular/translation rate 的中位数。

每个通过 coverage 的 window 只取第一个匹配角色：

1. `ROTATION_TRUTH_ELIGIBLE`：window angular `>=5 deg/s`、translation
   `<=0.02 m/s`、median over truth-covered pairs of
   `median_grid(abs(c_truth_grid)) <=0.02 s^-1`；
2. `STATIC_APPROACH_TRUTH_ELIGIBLE`：window truth closing `>=0.05 s^-1`；
   mixed rotation 允许，但 static consistency 必须通过；
3. `NOT_EVALUABLE_NO_FROZEN_PHASE_B_CONDITION`。

每个分支少于两个不同 sequence 时，该分支为
`NOT_EVALUABLE_INSUFFICIENT_SOURCE_NATIVE_TRUTH`。两分支都不足则终态
`HOLD_B1_SOURCE_NATIVE_TRUTH_NOT_EVALUABLE_NO_WINDOW_REPLACEMENT`；
至少一个分支足够则
`B1A_SOURCE_NATIVE_GEOMETRY_ADMISSION_VALID_B1B_BRANCH_SCOPE_MAY_BE_REVIEWED`。
B1A ledger/receipt canonical publish 后不可重跑、修补或替换。

## B1B：frozen ten-window metric audit

### 唯一执行与实现锁

canonical output：
`artifacts.local/evidence/rcle_phase_b_bonn_b1/b1b_metric_audit`。第一项
application data operation 的 claim 必须发生在 B1A ledger/receipt、RGB/ZIP、
project lock 或算法 module 的任何 read 前。claim 内嵌 expected B1A
ledger/receipt SHA、active branch exact enum 与 inactive window
dispositions。claim 固定为 canonical output 下永久 `run_claim.json`，不可
删除、替换或重写；其余 preclaim allowed/forbidden、无 override、异常消耗和
atomic publish 与 B1A 逐项相同。B1A PASS 只允许另立 B1B 设计/实现审查，
不自动执行。

B1B 只允许读取 frozen `VALID` B1A ledger/receipt、B0 receipt/window
denominator、固定 ZIP 中属于 B1A frozen active branch windows 的 candidate
pair 所引用 RGB PNG、hash-bound locks/config/algorithm sources。inactive
branch 不读 RGB、不运行 methods。禁止读取 depth PNG、`groundtruth.txt` 的
数值 pose、static map 或旧 Bonn outcome；禁止重算、修补或覆盖 B1A
truth/role。
B1B independent validator 使用同一 firewall，只重算 RGB methods/statistics，
不得重跑 B1A geometry。

B1B implementation lock 必须绑定：

- Phase A config
  `d20e77f3ea5f7ac55376006f1d14feb0ffb5daffd10a42792912fb89cdb1b502`
  及 lock
  `b9b9a51fb2ef3c1568cf573d0f00969948ac70d7b6825f1018d2e3aa7378820b`；
- rotation kernel `ab964e…ff24`、Sparse LK `36401e…73ca`、local affine
  `41e67c…8dbd`、support manager `83ac2e…7d08`；
- Observable Support implementation lock `a1dc13…5497`；
- Python/NumPy/OpenCV/Pillow exact versions、OpenCV threads `1`。

每个 source row rank 都是去除 comment/blank 后从零开始。pair `UNIT_ID`
的 UTF-8 bytes 精确为
`sequence_id + NUL + decimal(window_rank) + NUL +
decimal(previous_rgb_source_row_rank) + NUL +
decimal(current_rgb_source_row_rank)`；NUL 是单 byte `0x00`，decimal 无符号、
无前导零。seed input 是 ASCII/UTF-8 `RCLE_B1_RANSAC` 后一个 `0x00`，再接
UNIT_ID bytes。seed 为 SHA-256 前四 bytes 的 big-endian uint32 与
`0x7fffffff`。每个 pair 的 `BASELINE_M0`、`BASELINE_M1`、
`MANAGED_M0`、`MANAGED_M1` 每次完整九-grid fit invocation 前都重置同一
seed；arm/refit 名称不进入 seed。OpenCV `setNumThreads(1)`；禁止 runtime
参数覆盖。

### 唯一三个方法

RGB 统一由 Pillow `12.2.0` `Image.open(BytesIO(bytes)); load()` 解码；要求
`format=="PNG"`、`mode=="RGB"`、`size==(640,480)`，`np.asarray` 精确为
uint8 `(480,640,3)`；不允许 `convert()`。随后唯一灰度转换是 OpenCV
`cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)`。

1. `M0_RAW_LOCAL_EXPANSION`：上述灰度、Sparse LK、固定 `3x3` local
   affine，不 warp。
2. `M1_ROTATION_COMPENSATED_LOCAL_EXPANSION`：与 M0 共享相同 initial
   feature pool，只用 B1A `R_current_from_previous` 和上式 homography。
   sealed-validation PASS 的 `OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`
   必须按原设计对 raw 与 compensated 两臂对称应用，而不是只增强 M1：
   先对两臂各跑 baseline；任一臂 cell support `<12` 或 hull `<0.10` 就对
   两臂共同激活该 cell。baseline pool 是 raw/comp 任一路径 accepted 的
   previous points union，保持 detector initial-point order，以 float32
   coordinate exact tuple membership 去重。

   prior carry 请求顺序保持上一个 continuity-capable pair 的 survivor order；
   候选只限 activated cells，raw OR comp accepted 才进入 shared admission。
   以全部 baseline pool 为初始 occupied，按请求顺序处理；point 属于
   ascending activated-cell index 中第一个匹配 cell，该 cell occupied 已达
   80 或与任一全局 occupied 距离 `<5 px` 就拒绝，否则 admitted 并加入
   occupied。shared admitted mask 给两臂，但每臂自己的 accepted flag 决定
   是否进入该臂 correspondence。

   raw/comp 任一分类为 observable occlusion，就把同一个 prior endpoint 加入
   shared `10 px` supplement exclusion union。supplement 按 ascending
   activated cell、cell 内 row-major `4x4` subcell；每 subcell 最多一个
   Shi-Tomasi maximum，排序 response descending、y ascending、x ascending；
   与 baseline+admitted-carry+earlier supplements 距离 `>=5 px`、与
   exclusion `>=10 px`，每 cell 总 cap 80。相同 supplement points 给两臂，
   各自做 forward/FB/photometric acceptance。

   每臂 merge 顺序固定 baseline accepted、该臂 accepted carry、该臂 accepted
   supplement；activated cells 两臂都做 managed consensus/refit，未激活 cell
   保留 baseline。M0 正式输出是 activated cell 的 managed raw、其他 cell
   baseline raw；M1 对 comp 同理。next-pair prior survivor 只从
   `(baseline raw observable, raw carry, raw supplement)` 依此顺序取 accepted
   tracks；各类保持 source order，以 source point 与已保留点距离 `<5 px`
   去重，先到先留。

   raw validity 是 source image bounds；comp validity 是 previous bounds 与
   rotation-warp valid mask 交集。manager 状态每窗重置。只有相邻 candidate
   pairs 满足前一 pair current RGB rank 等于后一 pair previous RGB rank，且
   两 leg 的 RGB decode、dt、pose/homography、raw tracking/baseline observable
   和两臂 pipeline 都完整，才可消费 prior state。dt-outside gap、row 跳跃、
   decode/pose/homography/track failure 或 incomplete arm 都 reset；reset 后
   首个 continuity-capable pair 两臂 unchanged baseline。truth-uncovered 或
   metric-ineligible pair 若 pipeline 完整，可建立/延续 raw-leg state，但不
   进入 metric numerator。
3. `M2_FIXED_GRID_IMAGE_SCALE_PROXY`：只用 M0 raw path 中 shared initial
   pool 经 finite forward/backward `<=1 px` 接受的 raw tracks，不用 affine
   consensus/inliers、不 warp。只在同一 previous-frame grid 内取全部
   unordered point pairs；`d_previous>=20 px`、`d_current>0`、finite，
   grid rate 是至少 12 个有效 point-pair 的
   `median(log(d_current/d_previous)/dt)`。pair rate 是至少 `5/9` grid
   rates 的中位数；window M2 coverage 是 M2-evaluable candidate pairs /
   candidate denominator，门 `>=0.80`。M2 只诊断，不参与 M1/M0 gate，
   不得改用 bbox。

禁止 import/read 旧 Bonn producer/manifest/trace/truth/result、Phase A
evaluator/generator/summary；禁止 neural flow、detector、bbox、semantic ROI
或事后可视筛选。

### estimator、共同 grid 与覆盖

M0/M1 原样保持 support `>=12`、hull `>=0.10`、condition `<=1000`、
median residual `<=0.75 px/frame`。每个 metric pair 的唯一 grid 集合：

```text
metric_common_grids =
    B1A truth-eligible grids
    intersect M0 evaluable grids
    intersect M1 evaluable grids
```

至少 `5/9` 才是 metric-evaluable pair。window method coverage =
metric-evaluable candidate pairs / B1A candidate denominator，必须
`>=0.80`；否则在 truth 已足时为 `FAIL_REAL_SOURCE_METHOD_COVERAGE`。
truth absence 与 method failure 分开报告。

### window、sequence 与 bootstrap

rotation：

- pair `L_raw/L_comp`：metric-common grids 的 `abs(expansion)` 中位数；
- window `L_raw/L_comp`：metric-evaluable pairs 的 pair statistic 中位数；
- window `deltaL=L_raw-L_comp`；
- sequence `L_raw/L_comp`：分别为该角色所有 evaluable windows 的中位数；
- paired `window_deltaL=window_L_raw-window_L_comp`；
- gate/CI 使用的 `sequence_deltaL=median(window_deltaL)`，不得改写为两个
  sequence marginal medians 的差。

approach：

- pair `E_raw/E_comp`：metric-common grids 的
  `abs(expansion-c_truth_grid)` 中位数；
- window error：metric-evaluable pairs 的 pair error 中位数；
- `pair_comp_expansion=median(metric_common_grid expansion_comp)`；
- `window_comp_expansion=median(metric-evaluable pair_comp_expansion)`；
- `sequence_comp_expansion=median(eligible window_comp_expansion)`；
- sequence `E_raw/E_comp`：分别为 eligible window errors 的中位数；
- paired `window_error_delta=window_E_comp-window_E_raw`；
- gate/CI 使用的 `sequence_error_delta=median(window_error_delta)`，不得使用
  两个 marginal sequence errors 的差。

eligible sequence 固定按 upstream `sequence_rank` 升序进入数组。每个 gate
metric 先得到每个 eligible sequence 的一个 statistic，point
estimate 是这些 sequence statistics 的中位数。cluster bootstrap 使用
NumPy `Generator(PCG64(20260726))`；对每个 metric 独立重建 RNG，抽取
`N` 个 sequence index（`N` 为该分支 eligible distinct sequences）且允许
重复，replicate 是抽中 sequence statistics（重复保留）的中位数；共 10000
次。95% CI 用 `numpy.quantile([0.025,0.975], method="linear")`。不得丢弃
replicate；任一 non-finite 或少于两个 sequence 即该分支不可评价。

rotation 分支同时要求：

- 至少两个不同 sequence；
- 每个 sequence median `deltaL>0`；
- `deltaL` CI lower `>0`；
- sequence-equal `deltaL` point estimate `>=0.015 s^-1`；
- sequence `L_comp` statistic 的 CI upper `<=0.040 s^-1`。

approach 分支同时要求：

- 至少两个不同 sequence；
- 每个 sequence compensated expansion statistic `>0`；
- sequence `E_comp` statistic 的 CI upper `<=0.050 s^-1`；
- sequence `(E_comp-E_raw)` statistic 的 CI upper `<=0.015 s^-1`；
- 每个 sequence `(E_comp-E_raw)<=0.015 s^-1`。

RSR/CRR 只作诊断。rotation window
`RSR=1-window_L_comp/window_L_raw`，仅当 window raw `>=0.03 s^-1`；
sequence RSR 是 evaluable window RSR 中位数。approach 对每个
metric-common grid 取 `max(expansion,0)`，再按 grid→pair→window 嵌套中位数
得到 `C_raw/C_comp`；仅当 window `C_raw>=0.03 s^-1` 时
`CRR=C_comp/C_raw`，sequence CRR 是 evaluable window CRR 中位数。低分母
或空集均 `NOT_EVALUABLE_DIAGNOSTIC_DENOMINATOR`，不得加 epsilon。
M2 coverage `<0.80` 只记 `NOT_EVALUABLE_M2_COVERAGE`，不触发 method
coverage terminal。

## Receipt、validator 与终态

B1A ledger 必须完整保存十 window rows、全部 adjacent/candidate pair rows、
每 pair 九个 grid dispositions、四个 truth counts、association、pose/role
contribution 和具名 abstention。B1B 保存同一十窗、全部 candidate pairs、
九 grid 的 B1A/M0/M1 intersection、M2 与 window/sequence contributions。

JSON UTF-8、object keys 排序；timestamps 保留 canonical Decimal string；
计算 float 必须 finite并以 Python `float.hex()` string 持久化；禁止 NaN/
Infinity。ledger identity 是 compact
`json.dumps(sort_keys=True,separators=(",",":"),ensure_ascii=False)` 的
SHA-256。schema、source-code/environment manifest 和 receipt 自身都锁定。

独立 validator 不 import producer。B1A validator 独立重写 association/
geometry/role；B1B validator 可以 import 同一 hash-bound 四个纯算法核，但
必须独立重写 orchestration、聚合和 gate。它从各自 firewall 允许的
hash-bound input 重算并核对
十窗 canonical keys、所有 disposition/count/decision、source/runtime hashes
及 canonical ledger identity。非 timing 连续值复算容差 `abs<=1e-12`，
整数/枚举/hash/gate/terminal exact；任一差异为 `INVALID`。timing 不进入
ledger identity，validator 只检查各 sample finite/nonnegative，并按
`numpy.quantile(method="linear")` 核对 mean/median/P95 字段，不要求重放
wall-clock 数值。validate-existing 只读且不创建第二 claim。

终态：

- 两分支均可评价且全部通过：
  `PHASE_B_B1_MINIMAL_REAL_SOURCE_AUDIT_PASS_KILL_GATE_B_MAY_BE_REVIEWED`
- 一分支通过、另一 source truth 不足：
  `PARTIAL_REAL_SOURCE_MECHANISM_EVIDENCE_ONLY_KILL_GATE_B_NOT_EVALUABLE`
- 任一可评价分支失败：
  `FAIL_KILL_GATE_B_NO_EXTENSION_NO_MODULE_STACKING`
- truth 已足但 method coverage 失败：
  `FAIL_REAL_SOURCE_METHOD_COVERAGE`
- authority/hash/firewall/denominator/receipt 漂移：
  `INVALID_EXECUTION_CLOSE_B1`

唯一决策优先级：

1. 任一 authority/hash/firewall/denominator/receipt violation：
   `INVALID_EXECUTION_CLOSE_B1`；
2. 任一 active B1B branch window 的 M0/M1 method coverage `<0.80`：
   `FAIL_REAL_SOURCE_METHOD_COVERAGE`；
3. 任一 active/evaluable branch 数值 gate 失败（包括仅有一个 active branch
   且它失败）：`FAIL_KILL_GATE_B_NO_EXTENSION_NO_MODULE_STACKING`；
4. 两个 branch active 且全 PASS：full PASS terminal；
5. 只有一个 branch active 且 PASS：partial mechanism terminal。

inactive branch window 完整保留为
`NOT_EVALUABLE_INSUFFICIENT_SOURCE_NATIVE_TRUTH`，不读 RGB、不运行 methods，
也不能触发 method-coverage terminal。

runtime 是 non-gating 诊断。对每个 active-branch candidate pair，只要 stage
开始就记录一个非负 ns sample；stage 固定为 `rgb_decode`,
`rotation_warp`, `sparse_lk`, `observable_support`, `local_affine`, `m2`,
`total_method`。失败只保留已开始 stage 的 sample，不排除 warmup。每 stage
summary 报 sample count、NumPy mean/median 与
`quantile(q=0.95,method="linear")`；validator 用 pair stage-start flags 核对
sample scope。若某 stage 的 sample count 为 `0`，mean/median/P95 全部固定
为 JSON `null`，disposition 固定为
`NOT_APPLICABLE_NO_STARTED_STAGE_SAMPLE`；只有 count 大于 `0` 时才计算三项，
且三项都必须是 finite nonnegative float。

失败后不得在十窗上调 association、truth、LK、support、阈值或叠加模块；
成功也只形成 Bonn real-source mechanism evidence，不证明动态人体独立轨迹、
视障用户、主动告警、安全或生产有效性。

## 设计审查门

审查必须确认 machine lock 与本文 parity、range truth、pose/homography 方向、
source hashes、join、pixel domain、四计数、嵌套分母、M2、bootstrap、
claim-first、atomic publish、独立 validator 和终态全部唯一。任一 `FAIL`
时不得实现；PASS 后只开放 B1A 实现，不开放 B1A canonical execution 或
任何 B1B action。
