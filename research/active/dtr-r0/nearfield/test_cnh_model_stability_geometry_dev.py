"""Small CPU-only stability geometry contracts, without generating the batch."""
import copy
import sys
import unittest
import numpy as np
import cnh_model_stability_geometry_dev as M


class StabilityGeometryTest(unittest.TestCase):
    def test_period_mirror_and_height_only(self):
        self.assertEqual(len(M.UNITS),48)
        self.assertEqual({u%12 for u in M.UNITS},set(range(12)))
        scene=dict(boxes=[dict(lo=[.2,-.1,1],hi=[.4,.26,1.2],rho=.5),
                          dict(lo=[2,-1,1],hi=[3,1,2],rho=.2)])
        before=copy.deepcopy(scene)
        for unit in (410004,410007):
            a,b=M.physical_boxes(scene,unit,'original'),M.physical_boxes(scene,unit,'HB')
            b[0]['lo'][1],b[0]['hi'][1]=a[0]['lo'][1],a[0]['hi'][1]
            self.assertEqual(a,b)
            expected=[-.4,-.2] if (unit//3)%2 else [.2,.4]
            self.assertEqual([a[0]['lo'][0],a[0]['hi'][0]],expected)
        self.assertEqual(scene,before)

    def test_pose_schedule_noise_and_intact_error(self):
        # Original lightweight scene motion generator, no CP/torch import.
        for unit in (410004,410005,410006,410007,410008,410009):
            scene=M.S.make_scenes(unit)[0]
            error=np.arange(16,dtype=np.float32)-8
            a=M.poses_for(scene,unit,3,error);b=M.poses_for(scene,unit,3,error)
            np.testing.assert_array_equal(a['nn'],b['nn'])
            for f in range(16):np.testing.assert_allclose(a['qq'][f,:3,:3],M.rotation(error[f])@a['public_query'][f,:3,:3],atol=1e-12)
        self.assertNotIn('torch',sys.modules)
        pool=np.arange(80,dtype=np.float32).reshape(5,16)
        i,e=M.choose_error(pool,410004,3)
        np.testing.assert_array_equal(e,pool[i]);e[0]=-99
        self.assertNotEqual(pool[i,0],-99)

    def test_censored_clear_and_overlap_not_negative(self):
        travel=np.tile(np.eye(4),(16,1,1));travel[:,2,3]=np.arange(16)*.08
        boxes=[dict(lo=[1.2,-.1,5],hi=[1.4,.84,5.2],rho=.5)]
        clear=M.evaluate_geometry(boxes,travel)
        self.assertTrue(clear['valid']);self.assertTrue(clear['evaluable']);self.assertTrue(clear['control'])
        self.assertFalse(clear['covered']);self.assertIsNone(clear['deadline_index'])
        boxes.append(dict(lo=[1.25,.5,5.05],hi=[1.35,1,5.15],rho=.2))
        invalid=M.evaluate_geometry(boxes,travel)
        self.assertFalse(invalid['valid']);self.assertFalse(invalid['evaluable'])
        self.assertIsNone(invalid['contact']);self.assertIsNone(invalid['control'])
        self.assertEqual(invalid['target_background_overlaps'],[1])


if __name__=='__main__':unittest.main()
