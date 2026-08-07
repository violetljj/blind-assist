# riskseg_r0_evaluation

状态：$status

## 研究问题与版本

RISKSEG R0 评测视图物化与校验。协议版本 R0；遵循 docs/RESEARCH_GOVERNANCE.md。本 Module 只产生声明范围内的诊断或开发证据。

## 稳定 Interface

从仓库根目录调用本目录公开的 un_*、alidate_* 或入口脚本；输入必须是显式 manifest/fixture，缺失、版本不符或无法解码时 fail closed。

## 输出

只写入 artifacts.local/evidence/riskseg_r0_evaluation/ 或调用方显式指定的 artifacts.local/ 子目录；不写仓库根目录和正式 App 资产。

## 安全边界

不访问 protected confirmation outcome，不把 synthetic、model-generated、consumed 或单设备结果写成 unseen、产品、安全或默认 App authority。跨域调用只能经过稳定 root Adapter 或 esearch.common。

## 停止条件

达到最小判别实验、预算耗尽、输入权威缺失或重复失败即停止当前 evidence version；failure scope 最小化为 ITEM、BRANCH 或 EVIDENCE_VERSION，不自动关闭整个研究问题。

## 假设与规则质疑

候选必须说明 causal difference、expected information gain、falsifier、cost 和 selection reason。对 current/threshold 的质疑必须版本化，不得静默绕过。

## 失败资产复用

失败输出可按实际内容复用为 negative evidence、diagnostic、regression fixture、canary、counterexample 或 source characterization；不得重新包装为 unseen confirmation。

## 产物边界

运行产物必须位于 artifacts.local/，不提交数据集、模型、设备日志或大文件。
