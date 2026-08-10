# TARO O0R truth-only preflight

状态：`WILD_LAB / PREFLIGHT_LOCK_ONLY / EXECUTION_NOT_AUTHORIZED`

## 研究问题与版本

本 Module 只验证 `TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK`
是否完整绑定 O0R 合同选出的 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` parent、每个
parent 的三个 ARKitScenes source asset、离线复验 argv、环境、预算和未创建的 exclusive roots。
它不判定任何真实 factor、query、模型或科学结果。

## 稳定 Interface

```powershell
E:/codex-tools/bin/blindassist-python.cmd `
  scripts/research/taro_o0r_truth_preflight/validate_taro_o0r_truth_preflight_lock.py `
  --lock docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.json
```

validator 从 hash-bound source-and-adapter contract 重算 24-parent roster，再按三个冻结 URL
template 展开 72 个精确 HEAD target，并核对 canonical request-plan SHA-256。它还会核对所有
binding、资源上限、权限位和 exclusive-root absence。validator 无网络代码、无 artifact writer，
也不会读取 source payload。

## 输出

只向 stdout 输出 `VALID` 或逐条错误；不创建文件。未来 source/truth/head evidence 只能写入
preflight lock 冻结且当前不存在的 `artifacts.local/` roots，并需另立 execution authority。

## 安全边界

- preflight lock 本身不授权 HEAD、GET、Range 或任何网络访问；
- 不授权 source body 下载/打开、archive 枚举、truth materialization 或 uncertainty fit；
- 不授权 DepthART inference、factorial execution、training、Android/device、产品或 safety claim；
- 当前绑定的旧 Assistive Geometry receipt 不足以覆盖 TARO 的 24-parent body access，必须保持
  `INSUFFICIENT_FOR_24_PARENT_BODY_ACCESS`，直到新的 route-specific signed receipt 被另行绑定。

## 停止条件

任一 binding、roster、URL template、request-plan digest、预算、权限位或 root absence 漂移即
fail closed。本 Module 不自动修复或替换 parent，也不创建或消费 one-shot roots。

## 假设与规则质疑

HEAD target 的存在性和 Content-Length 仍是未执行的 future evidence；静态展开不能冒充远端可用性。
如果官方 URL 或授权范围变化，必须另立 lock 版本，不得在当前版本静默替换。

## 失败资产复用

静态 mutation case 可作为回归 fixture；未来 HEAD/source 失败只可作为 source availability
diagnostic，不得改 denominator、换 parent 或包装为科学负证据。
