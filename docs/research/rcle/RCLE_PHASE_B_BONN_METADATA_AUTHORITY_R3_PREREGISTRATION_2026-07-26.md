# RCLE Phase B Bonn Metadata Authority R3 预注册

状态：`DESIGN_FROZEN / EXECUTION_AUTHORIZED_IF_REVIEW_PASS`

日期：2026-07-26

## Bootstrap 边界

Python formal runner 启动前不可避免会由解释器读取 runner 自身和标准库。R3
把 preclaim authority 缩到一个 hash-bound 最小 bootstrap runner：

- 允许解释器读取该 runner 和 Python runtime/stdlib；
- runner preclaim 不 import 任何仓库 project module；
- 不使用 `Path.resolve`、`exists/stat/glob/listdir`；
- 不使用 argparse；只直接判定 `sys.argv` 是空或单一
  `--validate-existing`；
- formal 路径只用 `__file__`、`os.path.dirname/join/normpath` 做词法拼接；
- canonical output 由独立 setup 预创建；
- 第一项应用数据文件操作是内联
  `os.open(O_WRONLY|O_CREAT|O_EXCL)` claim；
- claim 落盘并 fsync 后，才延迟 import R3 authority module并读取任何
  lock/receipt/metadata/control/environment；
- 之后任一失败均保留 claim且不得重跑。

Validate-only 不创建 claim，可延迟 import 后只读 canonical receipt。

## 固定内容

R0/R1/R2 都保留为 diagnostic execution history。R3 固定采用同一
26-row denominator、9 条历史排除、6 条 cohort 与
`513b770d…ae86e`，不重选、不替换、不读取 payload。

## 终态

- PASS：`CANONICAL_METADATA_AUTHORITY_R3_PASS_FORMAL_PHASE_B_B0_READY`
- FAIL：`CLOSE_CANONICAL_METADATA_AUTHORITY_R3_NO_RERUN`

PASS 只使 frozen B0 acquisition/timestamp-inventory 协议 ready；不授权 Phase B
metrics、Replay、Android、人体、安全或生产。
