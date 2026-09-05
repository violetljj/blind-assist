import json
from pathlib import Path
import tempfile
import unittest
import prepare_dtr_final_detector_ledgers as prepare


class PreparationGateTest(unittest.TestCase):
    def test_failed_source_creates_no_detector_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=root/'seal/raw/FIT_ONLY';raw.mkdir(parents=True)
            (raw/'roster-source-gate.json').write_text(json.dumps({'status':'NOT_EVALUABLE'}))
            model=root/'model.pt';model.write_bytes(b'synthetic-unused')
            output=root/'new-output'
            with self.assertRaisesRegex(ValueError,'three_source_groups_required'):
                prepare.prepare(root/'seal',output,model)
            self.assertFalse(output.exists())

    def test_source_gate_alone_cannot_replace_four_sensor_join(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);raw=root/'seal/raw/FIT_ONLY';raw.mkdir(parents=True)
            (raw/'roster-source-gate.json').write_text(json.dumps({'status':'SOURCE_GATE_MET'}))
            (raw/'r1-joined-result.json').write_text(json.dumps({'status':'INCOMPLETE'}))
            model=root/'model.pt';model.write_bytes(b'synthetic-unused')
            output=root/'new-output'
            with self.assertRaisesRegex(ValueError,'four_sensor_join_required'):
                prepare.prepare(root/'seal',output,model)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
