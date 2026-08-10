# TARO O0R truth-only materializer runtime

状态：`WILD_LAB / IMPLEMENTATION_LOCK_PASS / SYNTHETIC_ONLY / HEAD_NOT_RUN / SOURCE_UNOPENED / EXECUTION_NOT_AUTHORIZED`

## 研究问题与版本

本 Module 实现已冻结 ARKitScenes O0R truth-only 前门的 source I/O seam：精确 72-URL HEAD
plan、Content-Length receipt、bounded download、ZIP/member SHA+CRC、exact Decimal timestamp、
exact-stem K/pose、source-member→canonical-member provenance、24-parent truth materialization、
8-parent uncertainty fit、每 query 独立且 source-bound 的 confidence/range、9-query FARO truth bundle，
content-addressed ndarray reload gate，以及 root-creation-consumes-one-shot 的原子证据写入。

## 稳定 Interface

未来 HEAD-only runner 只接受一个参数：

```powershell
E:/codex-tools/bin/blindassist-python.cmd `
  scripts/research/taro_o0r_truth_materializer_runtime/run_head_preflight.py `
  --execution-lock <future-exact-head-execution-lock.json>
```

未来 source/truth-only runner 同样只接受一个参数：

```powershell
E:/codex-tools/bin/blindassist-python.cmd `
  scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py `
  --execution-lock <future-exact-truth-only-execution-lock.json>
```

当前两种 execution lock 均不存在；调用 runner 必须 fail closed。库函数允许 synthetic tests 注入 fake HEAD/GET
transport，但 production transport 只接受 preflight lock 展开的精确 HTTPS URL，拒绝 redirect、响应体超长、
Content-Length 漂移、压缩编码、复用/覆盖与 path traversal。

## 输出

未来 HEAD-only evidence 固定写入 `artifacts.local/evidence/taro/o0r-arkitscenes-head-r0/`；未来正式
source/truth execution 只能写入 preflight lock 冻结的 `artifacts.local/datasets/taro/`、
`artifacts.local/work/taro/` 与 `artifacts.local/evidence/taro/` 子根。truth evidence root 首次
原子创建即消费 R0，无论 PASS、FAIL 或 timeout；不得恢复、覆盖、删除或重跑。O0R factor-headroom root
在整个 truth-only 阶段保持不存在。

## 安全边界

- 当前只授权实现与 synthetic tests；没有发送 HEAD/GET，没有打开 24 个 selected source body；
- old B0 materializer 的 binary-float timestamp、nearest intrinsics 和 three-band truth 接口不得导入；
- official archive 的 `{video_id}_{timestamp}` member 必须保留 original path/SHA/CRC，并显式映射到
  adapter 所需的 canonical `{timestamp}` member；禁止用静默重命名丢失 provenance；
- 8-parent uncertainty model 必须在任何 eval payload decode 前，用全部 exact fit frame 一次性封存；
  每个 query 的 lookup 只能从 bound FARO/confidence 与该 query corridor 推导，禁止 caller scalar 与
  9-query frame scalar 复用；
- factor/uncertainty artifact 中的 ndarray 采用 canonical little-endian gzip blob；写后必须 hydrate 并
  重算回 in-memory adapter canonical SHA，单独保存 ndarray hash receipt 不构成持久化；
- ADAPTER_FIT 只产生 residual uncertainty，O0R_EVAL_CANDIDATE 才产生 FARO truth/query bundle；
- 不包含 DepthART、candidate-relative scale、factorial、training、Android/device、产品或 safety 路径。

## 停止条件

授权/binding/HEAD/Content-Length/hash/CRC/member/schema/timestamp/K/pose/roster/root/预算任一漂移即 fail
closed。`47333152` trajectory 若未来 HEAD 非 `200 + Content-Length`，R0 必须 NOT_EVALUABLE；不得换
parent。undefined denominator 必须 FAIL，不能 drop。

## 假设与规则质疑

25/25 focused tests 通过只说明 source→receipt→truth mechanics、one-shot 消费后故障记账与受信任 `artifacts.local` junction containment 可复验，不说明 72 个远端 asset
存在、真实 truth gate 可通过、因果 headroom 成立或手持 source 可代表穿戴式观察。

## 失败资产复用

合成 archive、fake transport 和 mutation case 可作回归 fixture。未来消费后的 source/evidence 只可作为
失败诊断和 source characterization，不得改 roster、denominator、gate 或包装为 unseen confirmation。
