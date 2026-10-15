# 六张原图的模型辅助目视诊断

2026-10-15。主两臂风险模型和分数均已封存、主评价完成后，主代理用图像工具
逐张查看以下六张原图。选择来自固定hash争议队列及两臂读出变化；这是有目的
的诊断样本，不是随机正确率估计，不是独立人工标注。原标签和主分数均未改。

原图在`artifacts.local/work/assistive-thesis-screen-20261015/accessible_qa/images/`；
完整预测、required标签与选择原因在本次payload的`evaluation/final_audit/`。

| 原图后缀 | 实际看到的内容与模型输出 | 对当前实验的含义 |
| --- | --- | --- |
| 00010954 | 瓶身清楚写着Acetaminophen、Extra Strength、equate。两模型product答Acetaminophen，而product标签为medicine；variety标签同时要求acetaminophen与extra strength，模型把前者放到了product | 具体品名与泛化类别、跨字段分配混在“错误”里；不能全部解释为看不清或误读 |
| 00014955 | 包装实体倒置、左侧被裁切，反光。可见Be Enchanted字样、body cream及部分Bath & Body Works/Signature Collection。输出把Enchant、Signature Coll、body cream分别置入product/brand/variety | 存在字段归属混淆；朝向、裁切与反光是可见输入问题 |
| 00012716 | 实体倒置的Kool-Aid小袋，整体模糊，品牌大字与Cherry仍可辨。product标签要求drink mix，模型给Kool-Aid | 品牌/类别混淆与类别词面评分共同影响；并非单纯整体无内容 |
| 00012312 | Kirkland罐头、Green Beans大字可见，顶部Signature小字区域较模糊且反光。第二臂把Kirkland扩为Kirkland Signature | 对自动必需短语评分是唯一救回；这次目视不能认证每个小字都清晰可读 |
| 00001404 | 近距离截断的麦片盒、文字模糊，RIGHT与碗中麦片可见。第一臂product=cereal，第二臂=EATING RIGHT | 加OCR后把品牌性文字放进商品类别，产生新增字段错误；附加文字并不保证更好 |
| 00019585 | 实体倒置的LEAN CUISINE包装，Fiesta Grilled Chicken可见，另有Discover the goodness宣传文字。brand两臂为Discover This/Fiesta，均偏离品牌；第二臂variety漏Fiesta | 可读文字被当错角色，倒置可能影响读出。字段支持命中不能证明字段归属正确 |

这六图至少暴露两类不同问题：读到的文字被分配到错误字段，以及包装实体
朝向不利于模型读取。文件EXIF校正仅修复解码坐标，不会把所有实体文字变正。
因此在原计算余量内单独登记了全96图统一180度旋转诊断；不按这六图单独优化。

另一个纯文字后验诊断：把同图三个输出字段合并后，主两臂均有53/76个字段的
必需词被覆盖。单图模型的37个严格错误中14个词已在其他字段出现，OCR条件
模型为15/38。它只说明“文字出现在别处”，不证明标签归属或语义已正确；
因此保留原字段评分，不把53/76改写成读取准确率。
