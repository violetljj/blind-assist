# GRAIL M1 V1 Development Result

日期：2026-08-25（Asia/Hong_Kong）

状态：`DEVELOPMENT / TARGET_CENTERING_LEAK / ROUTE_REJECTED_BEFORE_FORMAL_TEST / TEST_ROSTER_UNOPENED / NO_M2`

V1 自动采集得到 train 273、dev 109 positives，dev 中 66 个有同类干扰。冻结 DINO pooled + local-token referent、B2 单点 head 与 GRAIL K-set head 的最终 Development 为：B0=`15/109`、B1=`19/109`、B2=`50/109`、GRAIL=`39/109` pose success；GRAIL wrong-target=`27/66`、absence false commit=`9/109`、permutation=`109/109`。

只读层级归因显示：若 oracle 选中目标 candidate，GRAIL pose head 成功 `88/109`；referent 实际只选中 `57/109`，选择与 pose 同时成功为 `50/109`。但该数值不能进入 formal：collector 把每个 query 相机 yaw 直接设为朝向目标，B2 可以忽略 goal/reference 并利用“目标居中/正前方”泄漏，实际达到 `50/109`，高于 GRAIL `39/109`。

因此 V1 不运行已经冻结但从未采集的 test roster，不声称算法正/负结果。唯一 successor V2 改 query 信息源：在目标仍可见的 `target yaw + {-60,-30,0,30,60}` 中按 sample hash 排序选择，而不是目标居中；新 test salt 排除 V1 test roster。V2 的 pose uplift 必须相对 `max(B0,B1,B2)`，不再把 B2 排除在“最强简单 baseline”之外。
