"""Exercise the real capture tick until its first depth submission, without UE."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from ue_settling import settling_count


class DepthSubmitted(BaseException):
    pass


class InitialSettlingTests(unittest.TestCase):
    def run_case(self, filename, startup, index, callbacks, rgb_count):
        source = Path(__file__).with_name(filename).read_text()
        tick = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'tick')
        counts = dict(rgb=0, depth=0, prepare=0)

        def rgb_capture():
            counts['rgb'] += 1

        def depth_capture():
            counts['depth'] += 1
            # Stop before file/native export; BaseException bypasses the real
            # tick's failure handler, leaving all preceding branches unchanged.
            raise DepthSubmitted()

        class Output:
            def __truediv__(self, other): return self
            def exists(self): return False

        ns = dict(finished=False, after=0, stage=2, world=None, index=index,
                  warm=0, first_prepared=True, frame_started=None, report={},
                  started=0, OUT=Output(), pair_exporter=None, rgb_exporter=None,
                  cases=[{'settling_frames': 8}, {'settling_frames': 8}],
                  STARTUP=startup, settling_count=settling_count,
                  time=NS(monotonic=lambda: 1, perf_counter=lambda: 1),
                  os=NS(environ={'BA_UE_CADENCE':'burst', 'BA_UE_SETTLING':'full'}),
                  rgb=NS(capture_component2d=NS(capture_scene=rgb_capture)),
                  depth=NS(capture_component2d=NS(capture_scene=depth_capture)),
                  prepare=lambda case: counts.__setitem__('prepare', counts['prepare']+1))
        exec(compile(ast.Module(body=[tick], type_ignores=[]), filename, 'exec'), ns)
        for _ in range(callbacks-1):
            ns['tick'](0)
            self.assertEqual(counts['depth'], 0)
        with self.assertRaises(DepthSubmitted):
            ns['tick'](0)
        self.assertEqual(counts['rgb'], rgb_count)
        self.assertEqual(counts['depth'], 1)
        self.assertEqual(counts['prepare'], 0 if index == 0 else 1)

    def test_ready_first_frame_keeps_full_count_across_callbacks(self):
        for filename, count in (('grounding_capture.py',33), ('factorial_capture.py',51)):
            with self.subTest(filename=filename):
                self.run_case(filename,'ready',0,count,count)

    def test_fixed_initial_burst_unchanged(self):
        for filename, count in (('grounding_capture.py',33), ('factorial_capture.py',51)):
            with self.subTest(filename=filename):
                self.run_case(filename,'fixed',0,1,count)

    def test_ready_later_frame_still_bursts(self):
        for filename in ('grounding_capture.py','factorial_capture.py'):
            with self.subTest(filename=filename):
                self.run_case(filename,'ready',1,1,9)


if __name__ == '__main__': unittest.main()
