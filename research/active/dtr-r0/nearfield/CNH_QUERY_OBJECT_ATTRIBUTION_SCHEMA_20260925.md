# 后续采集的逐查询对象身份契约

本轮只定义 [v1 JSON Schema](cnh_query_object_attribution.schema.json)，不采集、不补标、不改写当前 train/dev，test 和 City 继续挂起。现存 `targets.npz` 只有六查询标签，不能从 clip 的插入物类别推断每个正例的成因。当前采集和 materializer **尚未输出这个新 sidecar**；下次获准采集前需要接入并验证生产端。

后续 evaluator 标签 sidecar 每帧一条，使用现有 `frame_key` 和源 manifest SHA-256 绑定，同步保存六个查询名。对象列表包含与查询盒发生闭合三角形接触的**所有**物理对象，而非仅最近可见目标或指定插入物。沿用 `cnh_street_e2e_materialize.physical_labels` 的几何及覆盖语义；不改变现有查询盒或标签定义。

- `object_id` 在同一来源版本、物理场地、布局内跨帧稳定。原生对象用 actor/component/instance 身份，插入物用冻结的 insertion 身份；必须加来源/场地/布局命名空间，不能用 mesh 路径、像素 ID 或每帧序号充当对象身份。`source_instance_key` 保留原始身份，`geometry_sha256` 绑定用于求交的几何描述（含变换）。对象身份不随几何哈希变化而变化。
- 同一查询中的 `object_id` 必须唯一；这个跨记录约束由未来生产端检查，JSON Schema 的 `uniqueItems` 仅能排除完全重复的对象记录。同一对象的身份与类别注释必须跨帧一致。
- 已完整覆盖的负例为 `label=0, objects=[]`。缺失覆盖且无已知接触为 `label=-1, objects=null`，不可用空数组替代 UNKNOWN。已有接触但覆盖不完整仍为正例，列表仅含已知对象，`attribution_status=PARTIAL`，不可宣称完整类别切片。
- `category` 和 `thin_rod` 来自可审计的对象目录/几何注释；未建立时用 `null`，不得猜测或将 `null` 当作非细杆。提供任一注释必须给出 `annotation_provenance`，引用目录版本及细杆的判定定义。当前 schema 不事后发明细杆尺寸阈值。
- 类别/细杆切片须报告完整身份和注释覆盖率；多对象查询允许多类别成员，分母可能重叠。未知、部分归因和缺注释单列。负例没有致因对象，不能按插入物类别强行划分类别假阳性。

身份、类别与细杆注释仅属于 evaluator 标签。不得写入模型输入的 `observations.npz`、特征或采样选择条件。接入时必须保留 native 三角形的 owner 到真实 instance 身份映射（现有辅助函数的 `paths` 是 mesh 资产路径，不能直接作为对象 ID），并对每个插入实例独立保留求交归属。未来接入验收应覆盖多对象接触、同 mesh 不同实例、跨帧身份稳定、已知负例与 UNKNOWN 区分、部分覆盖正例，以及观测/标签隔离。
