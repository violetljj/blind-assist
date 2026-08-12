# TARO O1R R11 Bounded Source Download Attempt 01 Pre-start Incident

状态：`PRESTART_FAILED_UNCONSUMED_SUPERSEDED / GET_0 / SOURCE_UNOPENED`

Attempt 01 在任何 source/evidence root 创建和任何 GET 之前停止。冻结 argv 使用直接脚本路径；在仓库根
以绑定 Python 运行该路径时，解释器在导入阶段报告 `ModuleNotFoundError: No module named 'scripts'`。
项目的稳定包入口是 `python -m scripts...`，不得通过未冻结的 `PYTHONPATH` 绕过。

因此本次 `GET/source-body/archive-decode/source-frame-decode/model/FARO/truth/training = 0`，两个 download
root 均不存在，one-shot 未消费。Attempt 01 锁保留且不得原地修改或重跑。

唯一允许的修正是将 argv 冻结为 `-m scripts.research.taro_o1r_r11_abstention_runtime.run_pool_download`，
增加入口回归测试、提交新的 implementation commit，再另立 Attempt 02 execution lock。数据 roster、144-row
request plan、HEAD evidence、字节/重试/deadline/evidence budget、authority 与 claim ceiling全部不变。
