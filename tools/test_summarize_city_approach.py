import unittest
from summarize_city_approach import sequence_summary,validate_label_identity


def row(d,status='ELIGIBLE',joint=False,alarm=False,clear=False):
    return dict(distance_m=d,eligibility=status,joint=joint,alert=alarm,clear_alert=clear,peak_hit=False)


class SequenceTests(unittest.TestCase):
    def test_nearest_first_hit_cannot_claim_persistence(self):
        r=sequence_summary([row(2),row(1,joint=True,alarm=True)])
        self.assertIsNone(r['all_nearer_eligible_joint']);self.assertEqual(r['nearer_eligible_count'],0)
        self.assertEqual(r['persistence'],'NOT_TESTED')

    def test_labels_reject_reordering_and_payload_changes(self):
        manifest=dict(sample_indices=[0,1]);receipt=dict(status='PASS',label_manifest_sha256='m',near_sha256='n',support_sha256='s',target_mask_sha256={'t':'a'})
        hashes=dict(manifest='m',near='n',support='s',targets={'t':'a'})
        validate_label_identity(manifest,receipt,[0,1],hashes)
        with self.assertRaises(ValueError):validate_label_identity(manifest,receipt,[1,0],hashes)
        with self.assertRaises(ValueError):validate_label_identity(manifest,receipt,[0,1],dict(hashes,near='changed'))

    def test_outside_and_unknown_not_false_negatives(self):
        r=sequence_summary([row(6,'VISIBLE_NO_ELIGIBLE_QUERY_SUPPORT',alarm=True),row(3,'UNKNOWN'),row(2)])
        self.assertEqual(r['eligible'],1);self.assertEqual(r['alert_misses'],1)
        self.assertEqual(r['farthest_sampled_query_alert_m'],6);self.assertIsNone(r['farthest_sampled_joint_m'])

    def test_first_hit_not_monotonic_threshold_and_unknown_kept(self):
        r=sequence_summary([row(3,joint=True,alarm=True),row(2,'UNKNOWN'),row(1)])
        self.assertEqual(r['farthest_sampled_joint_m'],3)
        self.assertEqual(r['nearer_joint_misses_m'],[1]);self.assertEqual(r['nearer_unknown_m'],[2])
        self.assertEqual(r['continuity'],'UNKNOWN');self.assertFalse(r['all_nearer_eligible_joint'])

    def test_background_alarm_not_paired_specific_hit(self):
        r=sequence_summary([row(2,joint=True,alarm=True,clear=True),row(1,joint=True,alarm=True)])
        self.assertEqual(r['joint_hits'],2);self.assertEqual(r['joint_with_clear_silent'],1)


if __name__=='__main__':unittest.main()
