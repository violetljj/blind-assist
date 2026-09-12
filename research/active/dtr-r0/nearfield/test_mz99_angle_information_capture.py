"""Provenance instrumentation preserves frozen source/RNG/noisy slot selection."""
import ast
from pathlib import Path
import random
import unittest

import mz96_ue_decision_capture as baseline
import mz99_angle_information_capture as candidate


class ProvenanceTests(unittest.TestCase):
    def test_source_distribution_unchanged(self):
        a=baseline.source(99013);b=candidate.source(99013)
        b['schema']=a['schema']
        for row in b['scenes']:row['episode_id']=row['episode_id'].replace('ue99-','ue96-')
        self.assertEqual(a,b)

    def test_all_rng_call_sites_unchanged(self):
        def calls(module):
            tree=ast.parse(Path(module.__file__).read_text())
            return [ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and isinstance(n.func.value,ast.Name) and n.func.value.id in ('rng','ghosts')]
        self.assertEqual(calls(baseline),calls(candidate))

    def test_measurement_and_rng_state_parity(self):
        a=random.Random(987);b=random.Random(987)
        for _ in range(100):
            self.assertEqual(baseline.measure_radar(3.4,18.,-.65,10.,a),candidate.measure_radar(3.4,18.,-.65,10.,b))
        self.assertEqual(a.getstate(),b.getstate())

    def test_original_stable_sort_and_truncation(self):
        radar=[(3.,10.,-.5),(1.,20.,-.6),(1.,20.,-.6),(2.,0.,-.8),(4.,-10.,-.2)]
        provenance=[dict(kind=str(i),exact_angle_deg=i) for i in range(5)]
        slots=candidate.retained_slots(radar,provenance)
        self.assertEqual([r for r,p in slots],sorted(radar)[:4])
        self.assertEqual([p['kind'] for r,p in slots],['1','2','3','0'])
        self.assertTrue(all(p['observed_triple']==list(r) for r,p in slots))

    def test_transient_has_no_correction_and_missing_origin_fails(self):
        slots=candidate.retained_slots([(2.,30.,-.5)],[dict(kind='transient',exact_angle_deg=None)])
        self.assertIsNone(slots[0][1]['exact_angle_deg'])
        self.assertEqual(slots[0][1]['observed_triple'][1],30.)
        with self.assertRaises(ValueError):candidate.retained_slots([(2.,30.,-.5)],[])


if __name__=='__main__':unittest.main()
