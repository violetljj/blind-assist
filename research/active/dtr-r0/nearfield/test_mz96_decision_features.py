import unittest
import numpy as np
import mz96_decision_features as m
import mz90_observation_source as source

class FeatureChecks(unittest.TestCase):
    def test_prefix_permutation_and_schema(self):
        raw=source.materialize(source.build_source()[:3],'sensor_proxy')[0]
        full=m.build(raw);part=m.build({k:v[:65] for k,v in raw.items()})
        shuffled=m.build({k:v[:,::-1] if k.startswith('radar_') and v.ndim==2 else v for k,v in raw.items()})
        np.testing.assert_allclose(full[1][:65],part[1],equal_nan=True)
        np.testing.assert_allclose(full[1],shuffled[1],equal_nan=True)
        self.assertFalse(any(any(word in name for word in ('truth','family','episode','split','ghost')) for name in full[2]))
        for name in full[3]:np.testing.assert_array_equal(full[3][name][:65],part[3][name])

if __name__=='__main__':unittest.main()
