import unittest
import dtr_final_prediction_stages as stages
import dtr_final_structural_prediction_adapter as structural
import dtr_final_classic_prediction_adapter as classical
import test_dtr_final_structural_prediction_adapter as structural_fixture


class PredictionStagesTest(unittest.TestCase):
    def test_fit_rejects_final_and_reference_rosters_before_training(self):
        with self.assertRaises(ValueError):stages.fit_models('FINAL_A',{}, {},{})
        with self.assertRaises(ValueError):stages.fit_models('FIT_ONLY',{}, {},{'strata':{}})

    def test_alignment_rejects_silent_zip_truncation_and_time_drift(self):
        row={'sample_index':1,'time_s':.1}
        with self.assertRaises(ValueError):stages.align([row],[])
        with self.assertRaises(ValueError):stages.align([row],[dict(row,time_s=.2)])

    def test_real_empty_causal_chain_merges_nine_and_roundtrips_two_models(self):
        lock=structural.dependency_manifest()
        structural._load_locked(lock)
        ep,calibration=structural_fixture.StructuralAdapterTests().synthetic()
        candidates=[{'candidates':[]} for _ in ep.observations]
        raw=stages.raw.predict_episode(ep,candidates,calibration)
        classic=classical.predict_episode(ep,candidates,calibration)
        ours=structural.predict_episode(ep,candidates,calibration,dependency_lock=lock)
        merged=stages.merge_nonlearned(raw,classic,ours)
        self.assertEqual(9,len(merged['frames'][0]['arms']))
        tiny=stages.classic.TinyLogisticModel(stages.np.zeros(8),stages.np.ones(8),stages.np.zeros(8),0.)
        n=len(stages.x95.FEATURE_NAMES)
        hazard=stages.x95.LogisticEmission(stages.np.zeros(n),stages.np.ones(n),stages.np.zeros(n+1))
        learned=stages.apply_learned(merged,{'fit_group':'FIT_ONLY','tiny':tiny.to_json(),'x95':hazard.to_json()})
        self.assertEqual(11,len(learned['frames'][0]['arms']))
        self.assertTrue(learned['frames'][0]['arms']['TINY_LEARNED_PREDICTOR']['route_risk'])
        self.assertFalse(learned['frames'][0]['arms']['X95_EVENT_CHALLENGER']['route_risk'])
        self.assertEqual(9,len(merged['frames'][0]['arms']))


if __name__=='__main__':unittest.main()
