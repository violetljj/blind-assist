import unittest
import numpy as np
from cnh_street_e2e_train import layout_split, metrics


class DevelopmentTrainingTest(unittest.TestCase):
    def test_split_keeps_clips_together_and_balances_host_environment(self):
        rows = []
        for host in ('main', 'worker'):
            for environment in ('sidewalk', 'intersection', 'plaza'):
                for layout in range(2):
                    for clip in ('centre', 'boundary', 'outside', 'removed'):
                        rows.append(dict(machine_id=host, environment_category=environment,
                            layout_id=f'{host}-{environment}-{layout}', clip_id=clip, physical_site_id='shared'))
        train, val, report = layout_split(rows[::-1])
        self.assertEqual(int(train.sum()), 24)
        self.assertEqual(int(val.sum()), 24)
        self.assertFalse(set(report['train_layouts']) & set(report['debug_validation_layouts']))
        self.assertEqual(report['physical_sites'], ['shared'])
        for group in ('main', 'worker'):
            self.assertEqual(sum(x.startswith(group) for x in report['train_layouts']), 3)

    def test_unknown_is_excluded_not_negative(self):
        result = metrics(np.array([[-1, 1, 0, 1, 0, -1]]), np.array([[100, 1, 1, -1, -1, -100]]))
        self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']), (1, 1, 1, 1))
        self.assertEqual(result['known_queries'], 4)
        self.assertEqual(result['unknown_queries'], 2)


if __name__ == '__main__':
    unittest.main()
