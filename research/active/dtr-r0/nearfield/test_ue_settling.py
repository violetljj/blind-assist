import unittest
from ue_settling import settling_count


class SettlingTests(unittest.TestCase):
    def test_only_exact_state_within_clip_reuses(self):
        c = dict(clip_id='clip', camera={'x': 1}, objects=[{'material': 'a'}])
        self.assertEqual(settling_count(c,c,0,policy='reuse'),32)
        self.assertEqual(settling_count(c,c,1,policy='reuse'),0)
        self.assertEqual(settling_count(c,c,1),8)
        for change in [dict(clip_id='new'),dict(camera={'x':2}),dict(objects=[{'material':'b'}])]:
            self.assertEqual(settling_count(dict(c,**change),c,1,policy='reuse'),8)

    def test_missing_state_does_not_reuse(self):
        self.assertEqual(settling_count({}, {}, 1, default=26, policy='reuse'),26)


if __name__ == '__main__':
    unittest.main()
