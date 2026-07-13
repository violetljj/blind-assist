-- Reviewed snapshot rows for the SANPO next-leap diagnostic report.
-- Upstream evidence:
--   test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713/training_manifest.jsonl
--   test-artifacts.local/segmentation-candidate/v6-a100-384-two-stage-probe-20260713/
--   test-artifacts.local/segmentation-candidate/v6-a100-384-seeds12-13-20260713/
--   test-artifacts.local/segmentation-candidate/v5-a100-full-20260713/
-- Values were reproduced from canonical masks, gate reports and read-only inference.

WITH split_class_share(class_name, split_name, share, pixels, frames) AS (
    VALUES
        ('walkable', 'Train', 0.53065, 582090518, 400),
        ('walkable', 'Dev', 0.37647, 11102532, 200),
        ('boundary / step / curb', 'Train', 0.00857, 9401225, 400),
        ('boundary / step / curb', 'Dev', 0.16976, 5006457, 200),
        ('obstacle', 'Train', 0.22336, 245012670, 400),
        ('obstacle', 'Dev', 0.19476, 5744040, 200),
        ('unknown', 'Train', 0.23742, 260429987, 400),
        ('unknown', 'Dev', 0.25902, 7638171, 200)
)
SELECT * FROM split_class_share;

WITH seed_performance(configuration_seed, resolution, seed, metric, value, selection_score) AS (
    VALUES
        ('256 / seed 11', 256, '20260711', 'mIoU', 0.3689, 0.2477),
        ('256 / seed 11', 256, '20260711', 'Boundary IoU', 0.1864, 0.2477),
        ('256 / seed 12', 256, '20260712', 'mIoU', 0.2311, 0.1031),
        ('256 / seed 12', 256, '20260712', 'Boundary IoU', 0.0663, 0.1031),
        ('256 / seed 13', 256, '20260713', 'mIoU', 0.2682, 0.0632),
        ('256 / seed 13', 256, '20260713', 'Boundary IoU', 0.0358, 0.0632),
        ('384 / seed 11', 384, '20260711', 'mIoU', 0.4344, 0.4424),
        ('384 / seed 11', 384, '20260711', 'Boundary IoU', 0.4506, 0.4424),
        ('384 / seed 12', 384, '20260712', 'mIoU', 0.1804, 0.1769),
        ('384 / seed 12', 384, '20260712', 'Boundary IoU', 0.1734, 0.1769),
        ('384 / seed 13', 384, '20260713', 'mIoU', 0.2498, 0.1912),
        ('384 / seed 13', 384, '20260713', 'Boundary IoU', 0.1548, 0.1912)
)
SELECT * FROM seed_performance;

WITH scene_metrics(scene, miou, walkable, boundary, obstacle, unknown_class) AS (
    VALUES
        ('step / curb', 0.2680, 0.3362, 0.5788, 0.1539, 0.0031),
        ('center obstacle', 0.3433, 0.5086, 0.0000, 0.2144, 0.6504),
        ('lateral pedestrian / e-bike', 0.3434, 0.7684, 0.0479, 0.3346, 0.2228),
        ('parallel boundary', 0.3585, 0.6086, 0.1355, 0.2935, 0.3964)
)
SELECT * FROM scene_metrics;
