import unittest
import finalize_dtr_final_roster_join as join


class JoinScopeTest(unittest.TestCase):
    def fixture(self):
        legacy={'protocol_sha256':'p','checks':{**dict.fromkeys(join.REQUIRED,True),join.LEGACY:False,'cross_sensor_actual_replay_identical':True},
                'status':'DTR_CARLA_C2_RICH_MULTILAYOUT_SOURCE_NOT_EVALUABLE','sealed_evidence_manifest_sha256':'h'}
        source={'status':'SOURCE_GATE_MET','provenance':{'capture_protocol_sha256':'p',
            'historical_diagnostic':False,'verified_sensors':['instance','witness']},
            'source_semantics':{'status':'SOURCE_GATE_MET','strata':[{'status':'SOURCE_GATE_MET'} for _ in range(10)]}}
        return legacy,source

    def test_partial_semantics_do_not_require_complete_occlusion(self):
        legacy,source=self.fixture()
        self.assertEqual('DTR_R1_FOUR_SENSOR_SOURCE_COMPLETE',join.decision(legacy,source,'p')['status'])
        self.assertFalse(legacy['checks'][join.LEGACY])

    def test_replay_or_any_source_failure_is_never_waived(self):
        legacy,source=self.fixture();legacy['checks']['cross_sensor_actual_replay_identical']=False
        with self.assertRaises(ValueError):join.decision(legacy,source,'p')
        legacy,source=self.fixture();source['source_semantics']['strata'][8]['status']='NOT_EVALUABLE'
        with self.assertRaises(ValueError):join.decision(legacy,source,'p')


if __name__=='__main__':unittest.main()
